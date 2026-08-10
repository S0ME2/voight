"""Stage-oriented, observable model-level batching for the v1 pipeline."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.artifacts import ArtifactWriter
from app.config import MrzSettings
from app.documents.mrz import MrzProfile, crop_polygon, preprocess, reconstruct, select
from app.documents.passport_localization import page_corners_from_mrz_width
from app.imaging import order_corners, warp_to_size
from app.pipeline import RegionProfile, complete_profile, prepare_profile_from_detection

Token = dict[str, Any]


class QueueFullError(RuntimeError):
    pass


class ResourceExhaustedError(RuntimeError):
    pass


class InferenceGate:
    """Bound waiting requests and serialize each complete shared-model pipeline."""

    def __init__(self, queue_limit: int):
        if queue_limit <= 0:
            raise ValueError("queue_limit must be greater than zero")
        self._slots = threading.BoundedSemaphore(queue_limit)
        self._model_lock = threading.Lock()

    def __enter__(self):
        if not self._slots.acquire(blocking=False):
            raise QueueFullError("inference queue is full")
        try:
            self._model_lock.acquire()
        except BaseException:
            self._slots.release()
            raise
        return self

    def __exit__(self, exc_type, exc, traceback):
        self._model_lock.release()
        self._slots.release()


@dataclass(frozen=True)
class OcrSample:
    item_id: str
    image: np.ndarray


@dataclass(frozen=True)
class OcrBatchResult:
    tokens: dict[str, list[Token]]
    errors: dict[str, Exception]
    diagnostics: dict[str, Any]


def _chunks(values: Sequence[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


def _result_value(result: Any, key: str, default: Any = None) -> Any:
    try:
        return result[key]
    except (KeyError, TypeError):
        return getattr(result, key, default)


def _line_crop(image: np.ndarray, polygon: Any) -> tuple[np.ndarray, np.ndarray]:
    points = order_corners(np.asarray(polygon, dtype=np.float32).reshape(4, 2))
    width = max(
        1,
        int(round(max(np.linalg.norm(points[1] - points[0]), np.linalg.norm(points[2] - points[3])))),
    )
    height = max(
        1,
        int(round(max(np.linalg.norm(points[3] - points[0]), np.linalg.norm(points[2] - points[1])))),
    )
    return warp_to_size(image, points, width, height), points


def _pad_detection_batch(
    indexed_images: Sequence[tuple[int, np.ndarray]],
) -> list[tuple[int, np.ndarray]]:
    """Right/bottom pad variable images so Paddle can stack one real tensor."""
    max_height = max(image.shape[0] for _, image in indexed_images)
    max_width = max(image.shape[1] for _, image in indexed_images)
    return [
        (
            index,
            cv2.copyMakeBorder(
                image,
                0,
                max_height - image.shape[0],
                0,
                max_width - image.shape[1],
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            ),
        )
        for index, image in indexed_images
    ]


def _is_resource_error(error: BaseException) -> bool:
    message = str(error).lower()
    return isinstance(error, MemoryError) or any(
        marker in message
        for marker in ("out of memory", "resource exhausted", "cuda error", "cudnn_status_alloc_failed")
    )


def _finish_stage(stage: dict[str, Any]) -> None:
    calls = stage["calls"]
    stage["model_call_count"] = len(calls)
    stage["submitted_batch_sizes"] = [call["submitted_batch_size"] for call in calls]
    stage["tensor_batch_sizes"] = [call["tensor_batch_size"] for call in calls if "tensor_batch_size" in call]
    stage["failure_count"] = sum(call["failure_count"] for call in calls)
    stage["model_seconds"] = sum(call["model_seconds"] for call in calls)
    stage["wall_seconds"] = stage.get(
        "elapsed_wall_seconds",
        sum(call.get("wall_seconds", call["model_seconds"]) for call in calls),
    )


class BatchedOcr:
    """Run Paddle detection and recognition with prediction-time microbatches."""

    def __init__(
        self,
        detector: Any,
        recognizer: Any,
        *,
        detection_batch_size: int,
        recognition_batch_size: int,
    ):
        if detection_batch_size <= 0 or recognition_batch_size <= 0:
            raise ValueError("OCR batch sizes must be greater than zero")
        self.detector = detector
        self.recognizer = recognizer
        self.detection_batch_size = detection_batch_size
        self.recognition_batch_size = recognition_batch_size

    @staticmethod
    def _predict(model: Any, images: list[np.ndarray]) -> list[Any]:
        predict = getattr(model, "predict", model)
        return list(predict(input=images, batch_size=len(images)))

    def _predict_chunk(
        self,
        stage_name: str,
        model: Any,
        indexed_images: list[tuple[int, np.ndarray]],
        calls: list[dict[str, Any]],
    ) -> tuple[dict[int, Any], dict[int, Exception]]:
        started = time.perf_counter()
        size = len(indexed_images)
        try:
            values = self._predict(model, [image for _, image in indexed_images])
            if len(values) != size:
                raise ValueError(f"{stage_name} returned {len(values)} results for {size} inputs")
        except Exception as error:
            if _is_resource_error(error):
                raise ResourceExhaustedError(f"{stage_name} resource failure") from error
            calls.append(
                {
                    "submitted_batch_size": size,
                    "tensor_batch_size": size,
                    "failure_count": size,
                    "model_seconds": time.perf_counter() - started,
                    "wall_seconds": time.perf_counter() - started,
                    "error_type": type(error).__name__,
                }
            )
            return {}, {index: error for index, _ in indexed_images}
        elapsed = time.perf_counter() - started
        calls.append(
            {
                "submitted_batch_size": size,
                "tensor_batch_size": size,
                "failure_count": 0,
                "model_seconds": elapsed,
                "wall_seconds": elapsed,
            }
        )
        return {index: value for (index, _), value in zip(indexed_images, values)}, {}

    def run(self, samples: Sequence[OcrSample]) -> OcrBatchResult:
        ids = [sample.item_id for sample in samples]
        if len(ids) != len(set(ids)):
            raise ValueError("OCR sample item IDs must be unique")
        invalid = {
            index: ValueError("OCR input must be a non-empty NumPy image")
            for index, sample in enumerate(samples)
            if not isinstance(sample.image, np.ndarray)
            or sample.image.size == 0
            or sample.image.ndim not in (2, 3)
        }
        valid = [(index, sample.image) for index, sample in enumerate(samples) if index not in invalid]
        diagnostics = {
            "text_detection": {"configured_batch_size": self.detection_batch_size, "calls": []},
            "text_recognition": {
                "configured_batch_size": self.recognition_batch_size,
                "calls": [],
            },
            "line_crop_seconds": 0.0,
            "result_unpack_seconds": 0.0,
        }

        detection_results: dict[int, Any] = {}
        errors = dict(invalid)
        for chunk in _chunks(valid, self.detection_batch_size):
            results, failures = self._predict_chunk(
                "text detection",
                self.detector,
                _pad_detection_batch(chunk),
                diagnostics["text_detection"]["calls"],
            )
            detection_results.update(results)
            errors.update(failures)

        line_crop_started = time.perf_counter()
        lines: list[tuple[int, np.ndarray, np.ndarray]] = []
        for sample_index, result in detection_results.items():
            if error := _result_value(result, "error"):
                errors[sample_index] = ValueError(str(error))
                continue
            polygon_value = _result_value(result, "dt_polys", [])
            polygons = [] if polygon_value is None else list(polygon_value)
            polygons.sort(key=lambda polygon: (float(np.asarray(polygon)[:, 1].mean()), float(np.asarray(polygon)[:, 0].min())))
            for polygon in polygons:
                try:
                    polygon = np.asarray(polygon, dtype=np.float32)
                    height, width = samples[sample_index].image.shape[:2]
                    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
                    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
                    crop, ordered = _line_crop(samples[sample_index].image, polygon)
                except (cv2.error, TypeError, ValueError) as error:
                    errors[sample_index] = error
                    lines = [line for line in lines if line[0] != sample_index]
                    break
                lines.append((sample_index, crop, ordered))
        diagnostics["line_crop_seconds"] = time.perf_counter() - line_crop_started

        recognition_results: dict[int, Any] = {}
        recognition_errors: dict[int, Exception] = {}
        recognition_chunks = [
            list(chunk)
            for chunk in _chunks(
                [(index, line[1]) for index, line in enumerate(lines)],
                self.recognition_batch_size,
            )
        ]
        if hasattr(self.recognizer, "predict_chunks") and recognition_chunks:
            recognition_started = time.perf_counter()
            try:
                predictions = self.recognizer.predict_chunks(
                    [[image for _, image in chunk] for chunk in recognition_chunks]
                )
                if len(predictions) != len(recognition_chunks):
                    raise ValueError("text recognition returned an unexpected number of batches")
                for chunk, (values, model_seconds) in zip(recognition_chunks, predictions):
                    if len(values) != len(chunk):
                        raise ValueError(f"text recognition returned {len(values)} results for {len(chunk)} inputs")
                    diagnostics["text_recognition"]["calls"].append(
                        {
                            "submitted_batch_size": len(chunk),
                            "tensor_batch_size": len(chunk),
                            "failure_count": 0,
                            "model_seconds": model_seconds,
                            "wall_seconds": model_seconds,
                        }
                    )
                    recognition_results.update(
                        {index: value for (index, _), value in zip(chunk, values)}
                    )
            except Exception as error:
                if _is_resource_error(error):
                    raise ResourceExhaustedError("text recognition resource failure") from error
                for chunk in recognition_chunks:
                    diagnostics["text_recognition"]["calls"].append(
                        {
                            "submitted_batch_size": len(chunk),
                            "tensor_batch_size": len(chunk),
                            "failure_count": len(chunk),
                            "model_seconds": 0.0,
                            "wall_seconds": 0.0,
                            "error_type": type(error).__name__,
                        }
                    )
                    recognition_errors.update({index: error for index, _ in chunk})
            diagnostics["text_recognition"]["elapsed_wall_seconds"] = time.perf_counter() - recognition_started
        else:
            for chunk in recognition_chunks:
                results, failures = self._predict_chunk(
                    "text recognition", self.recognizer, chunk, diagnostics["text_recognition"]["calls"]
                )
                recognition_results.update(results)
                recognition_errors.update(failures)

        result_unpack_started = time.perf_counter()
        tokens_by_index: dict[int, list[Token]] = {
            index: [] for index in range(len(samples)) if index not in errors
        }
        for line_index, (sample_index, _, polygon) in enumerate(lines):
            if line_index in recognition_errors:
                errors.setdefault(sample_index, recognition_errors[line_index])
                tokens_by_index.pop(sample_index, None)
                continue
            if sample_index in errors:
                continue
            result = recognition_results[line_index]
            text = _result_value(result, "rec_text", "")
            if isinstance(text, (list, tuple)):
                text = text[0] if text else ""
            score = float(_result_value(result, "rec_score", 0.0))
            x1, y1 = polygon.min(axis=0)
            x2, y2 = polygon.max(axis=0)
            item_tokens = tokens_by_index[sample_index]
            item_tokens.append(
                {
                    "index": len(item_tokens),
                    "text": str(text).strip(),
                    "score": score,
                    "x1": float(x1),
                    "y1": float(y1),
                    "x2": float(x2),
                    "y2": float(y2),
                    "center_x": float((x1 + x2) / 2),
                    "center_y": float((y1 + y2) / 2),
                    "height": max(1.0, float(y2 - y1)),
                }
            )
        diagnostics["result_unpack_seconds"] = time.perf_counter() - result_unpack_started

        for stage in (diagnostics["text_detection"], diagnostics["text_recognition"]):
            _finish_stage(stage)
        return OcrBatchResult(
            tokens={samples[index].item_id: value for index, value in tokens_by_index.items()},
            errors={samples[index].item_id: error for index, error in errors.items()},
            diagnostics=diagnostics,
        )


@dataclass(frozen=True)
class ProfileBatchItem:
    item_id: str
    image: np.ndarray
    profile: RegionProfile
    localization_kind: str
    parse_fields: Callable[[dict[str, list[Token]]], tuple[dict[str, Any], dict[str, str]]]
    validate_fields: Callable[[dict[str, Any]], list[str]]
    artifacts: ArtifactWriter
    canonical_width: int
    canonical_height: int
    padding: int = 0
    min_overlap: float = 0.3
    passport_page_corners: Any | None = None
    mrz_profile: MrzProfile | None = None
    probe_mrz: bool = False


@dataclass(frozen=True)
class ProfileBatchOutcome:
    item_id: str
    result: tuple[dict[str, Any], dict[str, Any]] | None = None
    mrz_text: str | None = None
    mrz_detected: bool = False
    error: Exception | None = None


class ProfileBatchRunner:
    """Run grouped localization, shared OCR, then independent parsing in order."""

    def __init__(
        self,
        ocr: BatchedOcr,
        localizers: dict[str, Any],
        mrz_settings: MrzSettings,
        *,
        localization_batch_size: int,
        max_items: int,
        queue_limit: int,
    ):
        if localization_batch_size <= 0 or max_items <= 0:
            raise ValueError("batch limits must be greater than zero")
        self.ocr = ocr
        self.localizers = localizers
        self.mrz_settings = mrz_settings
        self.localization_batch_size = localization_batch_size
        self.max_items = max_items
        self.gate = InferenceGate(queue_limit)

    def _localize(
        self,
        groups: dict[str, list[tuple[str, np.ndarray]]],
    ) -> tuple[dict[tuple[str, str], dict[str, Any]], dict[str, Exception], dict[str, Any]]:
        results: dict[tuple[str, str], dict[str, Any]] = {}
        errors: dict[str, Exception] = {}
        stages: dict[str, Any] = {}
        for kind, jobs in groups.items():
            stage = {"configured_batch_size": self.localization_batch_size, "calls": []}
            localizer = self.localizers[kind]
            for chunk in _chunks(jobs, self.localization_batch_size):
                started = time.perf_counter()
                size = len(chunk)
                try:
                    values = list(localizer.predict_batch([image for _, image in chunk]))
                    if len(values) != size:
                        raise ValueError(f"{kind} localization returned {len(values)} results for {size} inputs")
                    if getattr(localizer, "last_tensor_batch_size", None) != size:
                        raise ValueError(f"{kind} localization did not construct a tensor batch of {size}")
                except Exception as error:
                    if _is_resource_error(error):
                        raise ResourceExhaustedError(f"{kind} localization resource failure") from error
                    errors.update((item_id, error) for item_id, _ in chunk)
                    failure_count = size
                else:
                    results.update(((item_id, kind), value) for (item_id, _), value in zip(chunk, values))
                    failure_count = 0
                elapsed = time.perf_counter() - started
                call = {
                    "submitted_batch_size": size,
                    "failure_count": failure_count,
                    "model_seconds": float(
                        getattr(localizer, "last_model_seconds", elapsed)
                        if failure_count == 0
                        else elapsed
                    ),
                    "wall_seconds": elapsed,
                }
                if failure_count == 0:
                    call["tensor_batch_size"] = localizer.last_tensor_batch_size
                stage["calls"].append(call)
            _finish_stage(stage)
            stages[kind] = stage
        return results, errors, stages

    def run(self, items: Sequence[ProfileBatchItem]) -> tuple[list[ProfileBatchOutcome], dict[str, Any]]:
        if len(items) > self.max_items:
            raise ValueError(f"batch contains {len(items)} items; maximum is {self.max_items}")
        ids = [item.item_id for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("profile batch item IDs must be unique")
        started_total = time.perf_counter()

        with self.gate:
            groups: dict[str, list[tuple[str, np.ndarray]]] = {}
            padded_images: dict[str, np.ndarray] = {}
            for item in items:
                if item.localization_kind not in self.localizers:
                    raise ValueError(f"unknown localization kind: {item.localization_kind}")
                if item.localization_kind == "docaligner":
                    padded = cv2.copyMakeBorder(
                        item.image,
                        item.padding,
                        item.padding,
                        item.padding,
                        item.padding,
                        cv2.BORDER_CONSTANT,
                        value=(0, 0, 0),
                    )
                    padded_images[item.item_id] = padded
                    groups.setdefault("docaligner", []).append((item.item_id, padded))
                else:
                    groups.setdefault(item.localization_kind, []).append((item.item_id, item.image))
                if item.probe_mrz and item.localization_kind != "mrz":
                    groups.setdefault("mrz", []).append((item.item_id, item.image))

            localization, errors, localization_diagnostics = self._localize(groups)
            prepared: dict[str, Any] = {}
            mrz_polygons: dict[str, np.ndarray] = {}
            preparation_started = time.perf_counter()
            for item in items:
                if item.item_id in errors:
                    continue
                try:
                    if item.localization_kind == "mrz":
                        mrz = np.asarray(localization[(item.item_id, "mrz")]["mrz_polygon"], dtype=np.float32).reshape(4, 2)
                        if item.passport_page_corners is None:
                            raise ValueError("MRZ localization requires passport page geometry")
                        corners = page_corners_from_mrz_width(mrz, item.passport_page_corners)
                        mrz_polygons[item.item_id] = mrz
                    else:
                        padded_corners = np.asarray(
                            localization[(item.item_id, "docaligner")]["corners"], dtype=np.float32
                        ).reshape(4, 2)
                        corners = padded_corners - item.padding
                        if item.probe_mrz:
                            polygon = np.asarray(localization[(item.item_id, "mrz")]["mrz_polygon"], dtype=np.float32)
                            if polygon.size == 8:
                                mrz_polygons[item.item_id] = polygon.reshape(4, 2)
                    prepared[item.item_id] = prepare_profile_from_detection(
                        item.image,
                        item.profile,
                        corners,
                        item.artifacts,
                        canonical_width=item.canonical_width,
                        canonical_height=item.canonical_height,
                        padding=item.padding,
                        padded=padded_images.get(item.item_id),
                        padded_corners=(corners + item.padding),
                        started_total=started_total,
                    )
                except (cv2.error, IndexError, KeyError, TypeError, ValueError) as error:
                    errors[item.item_id] = error
            preparation_seconds = time.perf_counter() - preparation_started

            ocr_samples = [
                OcrSample(f"visible:{item.item_id}", prepared[item.item_id].data_crop)
                for item in items
                if item.item_id in prepared
            ]
            mrz_crop_started = time.perf_counter()
            for item in items:
                if item.mrz_profile is None or item.item_id not in mrz_polygons or item.item_id in errors:
                    continue
                try:
                    crop, expanded = crop_polygon(
                        item.image, mrz_polygons[item.item_id], self.mrz_settings.polygon_padding_ratio
                    )
                    processed = preprocess(crop, self.mrz_settings.max_side, self.mrz_settings.contrast)
                    item.artifacts.save_json(
                        "mrz_polygons.json",
                        {"detected_polygon": mrz_polygons[item.item_id], "expanded_polygon": expanded},
                    )
                    item.artifacts.save_image("mrz_crop.jpg", crop)
                    item.artifacts.save_image("mrz_preprocessed.png", processed)
                    sample_id = f"mrz:{item.item_id}"
                    ocr_samples.append(OcrSample(sample_id, processed))
                except (cv2.error, TypeError, ValueError) as error:
                    errors[item.item_id] = error
            mrz_crop_seconds = time.perf_counter() - mrz_crop_started

            ocr_result = self.ocr.run(ocr_samples)
            outcomes_by_id: dict[str, ProfileBatchOutcome] = {}
            for sample_id, error in ocr_result.errors.items():
                errors[sample_id.split(":", 1)[1]] = error
            result_assembly_started = time.perf_counter()
            for item in items:
                if item.item_id in errors:
                    outcomes_by_id[item.item_id] = ProfileBatchOutcome(item.item_id, error=errors[item.item_id])
                    continue
                try:
                    result = complete_profile(
                        prepared[item.item_id],
                        ocr_result.tokens[f"visible:{item.item_id}"],
                        item.parse_fields,
                        item.validate_fields,
                        min_overlap=item.min_overlap,
                        ocr_seconds=None,
                    )
                    mrz_text = None
                    sample_id = f"mrz:{item.item_id}"
                    if item.mrz_profile is not None:
                        tokens = ocr_result.tokens.get(sample_id, [])
                        selected = select(reconstruct(tokens), item.mrz_profile.line_counts)
                        mrz_text = "\n".join(line.text for line in selected)
                    result[1]["timings"]["total_seconds"] = time.perf_counter() - started_total
                except (cv2.error, IndexError, KeyError, TypeError, ValueError) as error:
                    outcomes_by_id[item.item_id] = ProfileBatchOutcome(item.item_id, error=error)
                else:
                    outcomes_by_id[item.item_id] = ProfileBatchOutcome(
                        item.item_id,
                        result=result,
                        mrz_text=mrz_text,
                        mrz_detected=item.item_id in mrz_polygons,
                    )
            result_assembly_seconds = time.perf_counter() - result_assembly_started

        diagnostics = {
            "total_wall_seconds": time.perf_counter() - started_total,
            "localization": localization_diagnostics,
            "pipeline": {
                "document_preparation_seconds": preparation_seconds,
                "canonicalization_seconds": sum(value.timings.get("canonicalization_seconds", 0.0) for value in prepared.values()),
                "data_crop_seconds": sum(value.timings.get("data_crop_seconds", 0.0) for value in prepared.values()),
                "mrz_crop_preprocess_seconds": mrz_crop_seconds,
                "result_assembly_seconds": result_assembly_seconds,
            },
            **ocr_result.diagnostics,
        }
        return [outcomes_by_id[item.item_id] for item in items], diagnostics
