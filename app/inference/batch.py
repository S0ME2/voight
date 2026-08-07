"""Bounded, observable model-level batching for profile extraction."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.artifacts import ArtifactWriter
from app.imaging import order_corners, warp_to_size
from app.pipeline import (
    RegionProfile,
    complete_profile,
    prepare_profile,
)

Token = dict[str, Any]


class QueueFullError(RuntimeError):
    pass


class InferenceGate:
    """Bound waiting work and serialize access to shared model instances."""

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
    top = np.linalg.norm(points[1] - points[0])
    bottom = np.linalg.norm(points[2] - points[3])
    left = np.linalg.norm(points[3] - points[0])
    right = np.linalg.norm(points[2] - points[1])
    width = max(1, int(round(max(top, bottom))))
    height = max(1, int(round(max(left, right))))
    return warp_to_size(image, points, width, height), points


class BatchedOcr:
    """Use separate public detection/recognition models for true batching."""

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
        return list(predict(images))

    def _isolated_predict(
        self,
        stage: str,
        model: Any,
        indexed_images: list[tuple[int, np.ndarray]],
        configured_size: int,
        calls: list[dict[str, Any]],
    ) -> tuple[dict[int, Any], dict[int, Exception]]:
        if not indexed_images:
            return {}, {}
        started = time.perf_counter()
        call = {
            "requested_size": configured_size,
            "actual_tensor_batch_size": len(indexed_images),
        }
        try:
            values = self._predict(model, [image for _, image in indexed_images])
            if len(values) != len(indexed_images):
                raise ValueError(
                    f"{stage} returned {len(values)} results for {len(indexed_images)} inputs"
                )
        except MemoryError:
            raise
        except Exception as error:
            call["failed"] = True
            call["error_type"] = type(error).__name__
            call["seconds"] = time.perf_counter() - started
            calls.append(call)
            if len(indexed_images) == 1:
                return {}, {indexed_images[0][0]: error}
            middle = len(indexed_images) // 2
            left_results, left_errors = self._isolated_predict(
                stage, model, indexed_images[:middle], configured_size, calls
            )
            right_results, right_errors = self._isolated_predict(
                stage, model, indexed_images[middle:], configured_size, calls
            )
            return {**left_results, **right_results}, {**left_errors, **right_errors}
        call["failed"] = False
        call["seconds"] = time.perf_counter() - started
        calls.append(call)
        return {
            index: value for (index, _), value in zip(indexed_images, values)
        }, {}

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
        valid = [
            (index, sample.image)
            for index, sample in enumerate(samples)
            if index not in invalid
        ]
        diagnostics = {
            "text_detection": {
                "configured_batch_size": self.detection_batch_size,
                "calls": [],
            },
            "text_recognition": {
                "configured_batch_size": self.recognition_batch_size,
                "calls": [],
            },
        }

        detection_results: dict[int, Any] = {}
        detection_errors = dict(invalid)
        for chunk in _chunks(valid, self.detection_batch_size):
            results, errors = self._isolated_predict(
                "text detection",
                self.detector,
                list(chunk),
                self.detection_batch_size,
                diagnostics["text_detection"]["calls"],
            )
            detection_results.update(results)
            detection_errors.update(errors)

        lines: list[tuple[int, np.ndarray, np.ndarray]] = []
        for sample_index, result in detection_results.items():
            error = _result_value(result, "error")
            if error:
                detection_errors[sample_index] = ValueError(str(error))
                continue
            polygon_value = _result_value(result, "dt_polys", [])
            polygons = [] if polygon_value is None else list(polygon_value)
            polygons.sort(
                key=lambda polygon: (
                    float(np.asarray(polygon)[:, 1].mean()),
                    float(np.asarray(polygon)[:, 0].min()),
                )
            )
            for polygon in polygons:
                try:
                    crop, ordered = _line_crop(samples[sample_index].image, polygon)
                except (cv2.error, TypeError, ValueError) as error:
                    detection_errors[sample_index] = error
                    lines = [line for line in lines if line[0] != sample_index]
                    break
                lines.append((sample_index, crop, ordered))

        recognition_results: dict[int, Any] = {}
        recognition_errors: dict[int, Exception] = {}
        indexed_lines = [(index, line[1]) for index, line in enumerate(lines)]
        for chunk in _chunks(indexed_lines, self.recognition_batch_size):
            results, errors = self._isolated_predict(
                "text recognition",
                self.recognizer,
                list(chunk),
                self.recognition_batch_size,
                diagnostics["text_recognition"]["calls"],
            )
            recognition_results.update(results)
            recognition_errors.update(errors)

        tokens_by_index: dict[int, list[Token]] = {
            index: [] for index in range(len(samples)) if index not in detection_errors
        }
        for line_index, (sample_index, _, polygon) in enumerate(lines):
            if line_index in recognition_errors:
                detection_errors.setdefault(sample_index, recognition_errors[line_index])
                tokens_by_index.pop(sample_index, None)
                continue
            if sample_index in detection_errors:
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

        for stage in diagnostics.values():
            calls = stage["calls"]
            stage["model_call_count"] = len(calls)
            stage["actual_tensor_batch_sizes"] = [
                call["actual_tensor_batch_size"] for call in calls
            ]
            stage["failure_count"] = sum(bool(call["failed"]) for call in calls)
            stage["seconds"] = sum(float(call["seconds"]) for call in calls)
        return OcrBatchResult(
            tokens={
                samples[index].item_id: value
                for index, value in tokens_by_index.items()
            },
            errors={
                samples[index].item_id: error
                for index, error in detection_errors.items()
            },
            diagnostics=diagnostics,
        )


@dataclass(frozen=True)
class ProfileBatchItem:
    item_id: str
    image: np.ndarray
    profile: RegionProfile
    detect_document: Callable[[np.ndarray], Any]
    parse_fields: Callable[
        [dict[str, list[Token]]], tuple[dict[str, Any], dict[str, str]]
    ]
    validate_fields: Callable[[dict[str, Any]], list[str]]
    artifacts: ArtifactWriter
    canonical_width: int
    canonical_height: int
    padding: int = 0
    min_overlap: float = 0.3


@dataclass(frozen=True)
class ProfileBatchOutcome:
    item_id: str
    result: tuple[dict[str, Any], dict[str, Any]] | None = None
    error: Exception | None = None


class ProfileBatchRunner:
    """Prepare regions, share OCR model calls, then restore stable item order."""

    def __init__(
        self,
        ocr: BatchedOcr,
        *,
        localization_batch_size: int,
        max_items: int,
        queue_limit: int,
    ):
        if localization_batch_size <= 0 or max_items <= 0:
            raise ValueError("batch limits must be greater than zero")
        self.ocr = ocr
        self.localization_batch_size = localization_batch_size
        self.max_items = max_items
        self.gate = InferenceGate(queue_limit)

    def run(
        self, items: Sequence[ProfileBatchItem]
    ) -> tuple[list[ProfileBatchOutcome], dict[str, Any]]:
        if len(items) > self.max_items:
            raise ValueError(
                f"batch contains {len(items)} items; maximum is {self.max_items}"
            )
        ids = [item.item_id for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("profile batch item IDs must be unique")

        with self.gate:
            prepared: dict[str, Any] = {}
            errors: dict[str, Exception] = {}
            localization_calls = []
            for item in items:
                started = time.perf_counter()
                try:
                    prepared[item.item_id] = prepare_profile(
                        item.image,
                        item.profile,
                        item.detect_document,
                        item.artifacts,
                        canonical_width=item.canonical_width,
                        canonical_height=item.canonical_height,
                        padding=item.padding,
                    )
                except MemoryError:
                    raise
                except Exception as error:
                    errors[item.item_id] = error
                localization_calls.append(
                    {
                        "requested_size": self.localization_batch_size,
                        "actual_tensor_batch_size": 1,
                        "failed": item.item_id in errors,
                        "seconds": time.perf_counter() - started,
                    }
                )

            ocr_result = self.ocr.run(
                [
                    OcrSample(item.item_id, prepared[item.item_id].data_crop)
                    for item in items
                    if item.item_id in prepared
                ]
            )
            errors.update(ocr_result.errors)
            outcomes = []
            for item in items:
                if item.item_id in errors:
                    outcomes.append(
                        ProfileBatchOutcome(item.item_id, error=errors[item.item_id])
                    )
                    continue
                try:
                    result = complete_profile(
                        prepared[item.item_id],
                        ocr_result.tokens[item.item_id],
                        item.parse_fields,
                        item.validate_fields,
                        min_overlap=item.min_overlap,
                        ocr_seconds=(
                            ocr_result.diagnostics["text_detection"]["seconds"]
                            + ocr_result.diagnostics["text_recognition"]["seconds"]
                        ),
                    )
                except MemoryError:
                    raise
                except Exception as error:
                    outcomes.append(ProfileBatchOutcome(item.item_id, error=error))
                else:
                    outcomes.append(ProfileBatchOutcome(item.item_id, result=result))

        diagnostics = {
            "localization": {
                "batching_supported": False,
                "reason": "installed DocAligner/MRZScanner public wrappers accept one image",
                "configured_batch_size": self.localization_batch_size,
                "model_call_count": len(localization_calls),
                "actual_tensor_batch_sizes": [1 for _ in localization_calls],
                "failure_count": sum(
                    bool(call["failed"]) for call in localization_calls
                ),
                "seconds": sum(
                    float(call["seconds"]) for call in localization_calls
                ),
                "calls": localization_calls,
            },
            **ocr_result.diagnostics,
        }
        return outcomes, diagnostics
