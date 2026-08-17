"""Stage-oriented, observable model-level batching for the v1 pipeline."""

from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy
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
from app.inference.contracts import MrzRecognitionResult
from app.inference.packing import SequentialBatchPacker
from app.pipeline import RegionProfile, complete_profile, prepare_profile_from_detection
from app.roi import roi_for_point

Token = dict[str, Any]


class QueueFullError(RuntimeError):
    pass


class ResourceExhaustedError(RuntimeError):
    pass


@dataclass(frozen=True)
class OcrSample:
    item_id: str
    image: np.ndarray
    recognition_rois: dict[str, dict[str, float]] | None = None
    role: str | None = None


@dataclass(frozen=True)
class OcrBatchResult:
    tokens: dict[str, list[Token]]
    errors: dict[str, Exception]
    diagnostics: dict[str, Any]


def _chunks(values: Sequence[Any], size: int):
    for start in range(0, len(values), size):
        yield values[start : start + size]


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


def _shape_stats(images: Sequence[np.ndarray]) -> dict[str, float]:
    ratios = [image.shape[1] / max(1, image.shape[0]) for image in images]
    padded = max(ratios) * len(ratios)
    return {
        "min_aspect_ratio": min(ratios),
        "max_aspect_ratio": max(ratios),
        "mean_aspect_ratio": sum(ratios) / len(ratios),
        "padded_width_estimate": padded,
        "unpadded_width_estimate": sum(ratios),
        "padding_efficiency_estimate": sum(ratios) / padded if padded else 1.0,
    }


def _call_shape_stats(
    images: Sequence[np.ndarray],
    source_images: Sequence[np.ndarray],
    role: str,
    *,
    transformed: bool,
) -> dict[str, Any]:
    heights = [int(image.shape[0]) for image in source_images]
    widths = [int(image.shape[1]) for image in source_images]
    padded_area = sum(int(image.shape[0]) * int(image.shape[1]) for image in images)
    unpadded_area = sum(height * width for height, width in zip(heights, widths))
    result: dict[str, Any] = {
        "role": role,
        "actual_batch_size": len(images),
        "input_widths": widths,
        "input_heights": heights,
        "max_width": max(widths),
        "max_height": max(heights),
        "sum_unpadded_pixel_area": unpadded_area,
        "padded_tensor_pixel_area": padded_area if transformed else None,
        "padding_efficiency": unpadded_area / padded_area if transformed and padded_area else None,
        "shape_metric_source": "padded model input" if transformed else "source crops; backend transform unavailable",
    }
    if not transformed:
        ratios = [width / max(1, height) for width, height in zip(widths, heights)]
        padded_width = max(ratios) * len(ratios)
        result["recognition_width_padding_efficiency"] = sum(ratios) / padded_width if padded_width else 1.0
    return result


def _sample_role(sample: OcrSample) -> str:
    if sample.role in {"visible", "mrz"}:
        return sample.role
    prefix = sample.item_id.split(":", 1)[0]
    return prefix if prefix in {"visible", "mrz"} else "unknown"


def _sample_digest(image: np.ndarray) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def _text_digest(lines: Sequence[str]) -> str:
    return hashlib.sha256(json.dumps(list(lines), ensure_ascii=False).encode()).hexdigest()


def _finish_stage(stage: dict[str, Any]) -> None:
    calls = stage["calls"]
    stage["model_call_count"] = sum(len(call.get("tensor_batch_sizes", (call.get("tensor_batch_size"),))) for call in calls)
    stage["submitted_batch_sizes"] = [call["submitted_batch_size"] for call in calls]
    stage["tensor_batch_sizes"] = [
        size
        for call in calls
        for size in call.get("tensor_batch_sizes", (call.get("tensor_batch_size"),))
        if size is not None
    ]
    stage["attempt_failure_count"] = sum(call["failure_count"] for call in calls)
    stage.setdefault("failure_count", stage["attempt_failure_count"])
    stage.setdefault("retry_split_count", 0)
    stage.setdefault("isolated_failure_count", 0)
    stage["model_seconds"] = sum(call["model_seconds"] for call in calls)
    stage["wall_seconds"] = stage.get(
        "elapsed_wall_seconds",
        sum(call.get("wall_seconds", call["model_seconds"]) for call in calls),
    )


def _finish_role_summaries(stage: dict[str, Any]) -> None:
    by_role: dict[str, dict[str, float | int]] = {}
    for call in stage["calls"]:
        role = call["role"]
        target = by_role.setdefault(
            role,
            {"model_calls": 0, "tensor_batches": 0, "model_seconds": 0.0, "wall_seconds": 0.0},
        )
        target["model_calls"] += 1
        target["tensor_batches"] += len(call.get("tensor_batch_sizes", (call.get("tensor_batch_size"),)))
        target["model_seconds"] += call["model_seconds"]
        target["wall_seconds"] += call["wall_seconds"]
    stage["by_role"] = by_role


class BatchedOcr:
    """Run Paddle detection and recognition with prediction-time microbatches."""

    def __init__(
        self,
        detector: Any,
        recognizer: Any,
        *,
        detection_batch_size: int,
        recognition_batch_size: int,
        recognition_packer: Any | None = None,
        capture_inputs: bool = False,
    ):
        if detection_batch_size <= 0 or recognition_batch_size <= 0:
            raise ValueError("OCR batch sizes must be greater than zero")
        self.detector = detector
        self.recognizer = recognizer
        self.detection_batch_size = detection_batch_size
        self.recognition_batch_size = recognition_batch_size
        self.recognition_packer = recognition_packer or SequentialBatchPacker()
        self.capture_inputs = capture_inputs
        self.captured_inputs: dict[str, np.ndarray] = {}

    def _predict_chunk(
        self,
        stage_name: str,
        model: Any,
        indexed_images: list[tuple[int, np.ndarray]],
        calls: list[dict[str, Any]],
        stats: dict[str, int],
        roles: Sequence[str],
        source_images: Sequence[np.ndarray],
    ) -> tuple[dict[int, Any], dict[int, Exception]]:
        started = time.perf_counter()
        size = len(indexed_images)
        try:
            images = [image for _, image in indexed_images]
            values = list(
                model.detect_batch(images)
                if stage_name == "text detection"
                else model.recognize_batch(images)
            )
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
                    **_call_shape_stats(
                        [image for _, image in indexed_images],
                        source_images,
                        "mixed" if len(set(roles)) > 1 else roles[0],
                        transformed=stage_name == "text detection",
                    ),
                    **(_shape_stats([image for _, image in indexed_images]) if stage_name == "text recognition" else {}),
                }
            )
            if size == 1:
                stats["isolated_failure_count"] += 1
                return {}, {indexed_images[0][0]: error}
            stats["retry_split_count"] += 1
            midpoint = size // 2
            left_results, left_failures = self._predict_chunk(
                stage_name, model, indexed_images[:midpoint], calls, stats
                , roles[:midpoint], source_images[:midpoint]
            )
            right_results, right_failures = self._predict_chunk(
                stage_name, model, indexed_images[midpoint:], calls, stats
                , roles[midpoint:], source_images[midpoint:]
            )
            return left_results | right_results, left_failures | right_failures
        elapsed = time.perf_counter() - started
        model_seconds = getattr(model, "last_model_seconds", None) or elapsed
        calls.append(
            {
                "submitted_batch_size": size,
                "tensor_batch_size": size,
                "failure_count": 0,
                "model_seconds": model_seconds,
                "wall_seconds": elapsed,
                **_call_shape_stats(
                    [image for _, image in indexed_images],
                    source_images,
                    "mixed" if len(set(roles)) > 1 else roles[0],
                    transformed=stage_name == "text detection",
                ),
                **(_shape_stats([image for _, image in indexed_images]) if stage_name == "text recognition" else {}),
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
            "text_detection": {
                "configured_batch_size": self.detection_batch_size,
                "calls": [],
                "retry_split_count": 0,
                "isolated_failure_count": 0,
            },
            "text_recognition": {
                "configured_batch_size": self.recognition_batch_size,
                "packing_strategy": self.recognition_packer.name,
                "calls": [],
                "retry_split_count": 0,
                "isolated_failure_count": 0,
            },
            "line_filter": {
                "detected_line_count": 0,
                "recognition_candidate_count": 0,
                "filtered_before_recognition_count": 0,
                "samples": {},
            },
            "line_crop_seconds": 0.0,
            "result_unpack_seconds": 0.0,
            "sample_records": [
                {
                    "sample_id": sample.item_id,
                    "sample_role": _sample_role(sample),
                    "sample_index": index,
                    "crop_sha256": _sample_digest(sample.image),
                    "width": int(sample.image.shape[1]) if sample.image.ndim >= 2 else None,
                    "height": int(sample.image.shape[0]) if sample.image.ndim >= 1 else None,
                }
                for index, sample in enumerate(samples)
            ],
        }
        if self.capture_inputs:
            self.captured_inputs = {sample.item_id: sample.image.copy() for sample in samples}

        detection_results: dict[int, Any] = {}
        errors = dict(invalid)
        for chunk in _chunks(valid, self.detection_batch_size):
            results, failures = self._predict_chunk(
                "text detection",
                self.detector,
                _pad_detection_batch(chunk),
                diagnostics["text_detection"]["calls"],
                diagnostics["text_detection"],
                [_sample_role(samples[index]) for index, _ in chunk],
                [samples[index].image for index, _ in chunk],
            )
            detection_results.update(results)
            errors.update(failures)

        line_crop_started = time.perf_counter()
        lines: list[tuple[int, np.ndarray, np.ndarray]] = []
        line_crop_seconds_by_role = {"visible": 0.0, "mrz": 0.0}
        for sample_index, result in detection_results.items():
            if result.error:
                errors[sample_index] = ValueError(result.error)
                continue
            polygons = [region.polygon for region in result.regions]
            polygons.sort(key=lambda polygon: (float(np.asarray(polygon)[:, 1].mean()), float(np.asarray(polygon)[:, 0].min())))
            sample = samples[sample_index]
            counts = {"detected_line_count": len(polygons), "recognition_candidate_count": 0, "filtered_before_recognition_count": 0}
            diagnostics["line_filter"]["samples"][sample.item_id] = counts
            diagnostics["line_filter"]["detected_line_count"] += len(polygons)
            for polygon in polygons:
                try:
                    crop_started = time.perf_counter()
                    polygon = np.asarray(polygon, dtype=np.float32)
                    height, width = sample.image.shape[:2]
                    polygon[:, 0] = np.clip(polygon[:, 0], 0, width - 1)
                    polygon[:, 1] = np.clip(polygon[:, 1], 0, height - 1)
                    center_x, center_y = (polygon.min(axis=0) + polygon.max(axis=0)) / 2
                    if sample.recognition_rois is not None and roi_for_point(
                        sample.recognition_rois, width, height, float(center_x), float(center_y)
                    ) is None:
                        counts["filtered_before_recognition_count"] += 1
                        diagnostics["line_filter"]["filtered_before_recognition_count"] += 1
                        continue
                    crop, ordered = _line_crop(sample.image, polygon)
                    role = _sample_role(sample)
                    line_crop_seconds_by_role.setdefault(role, 0.0)
                    line_crop_seconds_by_role[role] += time.perf_counter() - crop_started
                except (cv2.error, TypeError, ValueError) as error:
                    errors[sample_index] = error
                    lines = [line for line in lines if line[0] != sample_index]
                    diagnostics["line_filter"]["recognition_candidate_count"] -= counts["recognition_candidate_count"]
                    counts["recognition_candidate_count"] = 0
                    dropped = len(polygons) - counts["filtered_before_recognition_count"]
                    counts["filtered_before_recognition_count"] += dropped
                    diagnostics["line_filter"]["filtered_before_recognition_count"] += dropped
                    break
                lines.append((sample_index, crop, ordered))
                counts["recognition_candidate_count"] += 1
                diagnostics["line_filter"]["recognition_candidate_count"] += 1
        diagnostics["line_crop_seconds"] = time.perf_counter() - line_crop_started
        diagnostics["line_crop_seconds_by_role"] = line_crop_seconds_by_role

        recognition_results: dict[int, Any] = {}
        recognition_errors: dict[int, Exception] = {}
        recognition_chunks = self.recognition_packer.pack(
            [(index, line[1]) for index, line in enumerate(lines)],
            self.recognition_batch_size,
        )
        recognition_started = time.perf_counter()
        for chunk in recognition_chunks:
            results, failures = self._predict_chunk(
                "text recognition", self.recognizer, chunk,
                diagnostics["text_recognition"]["calls"], diagnostics["text_recognition"],
                [_sample_role(samples[lines[line_index][0]]) for line_index, _ in chunk],
                [lines[line_index][1] for line_index, _ in chunk],
            )
            recognition_results.update(results)
            recognition_errors.update(failures)
        diagnostics["text_recognition"]["elapsed_wall_seconds"] = time.perf_counter() - recognition_started
        diagnostics["text_detection"]["failure_count"] = len(
            set(errors) & set(sample_index for sample_index, _ in valid)
        )
        diagnostics["text_recognition"]["failure_count"] = len(recognition_errors)
        diagnostics["text_recognition"]["original_index_restoration_success"] = (
            set(recognition_results) | set(recognition_errors) == set(range(len(lines)))
        )

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
            text = result.text
            score = float(result.score or 0.0)
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
            _finish_role_summaries(stage)
        diagnostics["sample_counts"] = {
            role: sum(_sample_role(sample) == role for sample in samples)
            for role in ("visible", "mrz")
        }
        diagnostics["line_counts_by_role"] = {
            role: {
                "detected": sum(
                    value["detected_line_count"]
                    for sample_id, value in diagnostics["line_filter"]["samples"].items()
                    if sample_id.startswith(role + ":")
                ),
                "recognized": sum(
                    value["recognition_candidate_count"]
                    for sample_id, value in diagnostics["line_filter"]["samples"].items()
                    if sample_id.startswith(role + ":")
                ),
            }
            for role in ("visible", "mrz")
        }
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
    mrz_fallback_for: str | None = None


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
        mrz_recognizer: Any | None = None,
        mrz_recognition_batch_size: int | None = None,
        ocr_grouping: str = "combined",
    ):
        if localization_batch_size <= 0 or max_items <= 0 or (
            mrz_recognition_batch_size is not None and mrz_recognition_batch_size <= 0
        ):
            raise ValueError("batch limits must be greater than zero")
        if ocr_grouping not in {"combined", "split"}:
            raise ValueError("ocr_grouping must be combined or split")
        self.ocr = ocr
        self.localizers = localizers
        self.mrz_settings = mrz_settings
        self.localization_batch_size = localization_batch_size
        self.mrz_recognizer = mrz_recognizer
        self.mrz_recognition_batch_size = mrz_recognition_batch_size or ocr.recognition_batch_size
        self.max_items = max_items
        self.ocr_grouping = ocr_grouping

    @staticmethod
    def _merge_ocr_results(results: Sequence[OcrBatchResult]) -> OcrBatchResult:
        if len(results) == 1:
            return results[0]
        merged = deepcopy(results[0].diagnostics)
        for result in results[1:]:
            for stage_name in ("text_detection", "text_recognition"):
                merged[stage_name]["calls"].extend(result.diagnostics[stage_name]["calls"])
            merged["line_filter"]["samples"].update(result.diagnostics["line_filter"]["samples"])
            merged["line_filter"]["detected_line_count"] += result.diagnostics["line_filter"]["detected_line_count"]
            merged["line_filter"]["recognition_candidate_count"] += result.diagnostics["line_filter"]["recognition_candidate_count"]
            merged["line_filter"]["filtered_before_recognition_count"] += result.diagnostics["line_filter"]["filtered_before_recognition_count"]
            merged["sample_records"].extend(result.diagnostics["sample_records"])
            for role in ("visible", "mrz"):
                merged["sample_counts"][role] += result.diagnostics["sample_counts"][role]
                for key in ("detected", "recognized"):
                    merged["line_counts_by_role"][role][key] += result.diagnostics["line_counts_by_role"][role][key]
                merged["line_crop_seconds_by_role"][role] += result.diagnostics["line_crop_seconds_by_role"][role]
            merged["line_crop_seconds"] += result.diagnostics["line_crop_seconds"]
            merged["result_unpack_seconds"] += result.diagnostics["result_unpack_seconds"]
        for stage_name in ("text_detection", "text_recognition"):
            _finish_stage(merged[stage_name])
            _finish_role_summaries(merged[stage_name])
        merged["line_filter"]["samples"] = dict(merged["line_filter"]["samples"])
        merged["ocr_grouping"] = "split"
        return OcrBatchResult(
            tokens={key: value for result in results for key, value in result.tokens.items()},
            errors={key: value for result in results for key, value in result.errors.items()},
            diagnostics=merged,
        )

    def _recognize_mrz_chunk(
        self,
        indexed_images: list[tuple[str, np.ndarray]],
        calls: list[dict[str, Any]],
        stats: dict[str, int],
    ) -> tuple[dict[str, MrzRecognitionResult], dict[str, Exception]]:
        started = time.perf_counter()
        size = len(indexed_images)
        try:
            values = list(self.mrz_recognizer.recognize_batch([image for _, image in indexed_images]))
            if len(values) != size:
                raise ValueError(f"MRZ recognition returned {len(values)} results for {size} inputs")
        except Exception as error:
            if _is_resource_error(error):
                raise ResourceExhaustedError("MRZ recognition resource failure") from error
            elapsed = time.perf_counter() - started
            calls.append({"submitted_batch_size": size, "tensor_batch_size": size, "failure_count": size, "model_seconds": elapsed, "wall_seconds": elapsed, "error_type": type(error).__name__})
            if size == 1:
                stats["isolated_failure_count"] += 1
                return {}, {indexed_images[0][0]: error}
            stats["retry_split_count"] += 1
            midpoint = size // 2
            left, left_errors = self._recognize_mrz_chunk(indexed_images[:midpoint], calls, stats)
            right, right_errors = self._recognize_mrz_chunk(indexed_images[midpoint:], calls, stats)
            return left | right, left_errors | right_errors
        elapsed = time.perf_counter() - started
        call = {
            "submitted_batch_size": size,
            "tensor_batch_size": getattr(self.mrz_recognizer, "last_tensor_batch_size", size),
            "failure_count": 0,
            "model_seconds": getattr(self.mrz_recognizer, "last_model_seconds", elapsed),
            "wall_seconds": elapsed,
        }
        if sizes := getattr(self.mrz_recognizer, "last_tensor_batch_sizes", None):
            call["tensor_batch_sizes"] = list(sizes)
        calls.append(call)
        return {item_id: value for (item_id, _), value in zip(indexed_images, values)}, {}

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
                    values = list(localizer.localize_batch([image for _, image in chunk]))
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

    @staticmethod
    def _merge_localization_diagnostics(*groups: dict[str, Any]) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for group in groups:
            for kind, stage in group.items():
                target = merged.setdefault(kind, {
                    "configured_batch_size": stage["configured_batch_size"],
                    "calls": [],
                })
                target["calls"].extend(stage["calls"])
        for stage in merged.values():
            _finish_stage(stage)
        return merged

    def run(self, items: Sequence[ProfileBatchItem]) -> tuple[list[ProfileBatchOutcome], dict[str, Any]]:
        if len(items) > self.max_items:
            raise ValueError(f"batch contains {len(items)} items; maximum is {self.max_items}")
        ids = [item.item_id for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError("profile batch item IDs must be unique")
        started_total = time.perf_counter()

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
        localization, errors, localization_diagnostics = self._localize(groups)
        primary_jobs = [
            (item.item_id, item.image)
            for item in items
            if item.probe_mrz and item.localization_kind != "mrz"
        ]
        primary, _primary_errors, primary_diagnostics = self._localize(
            {"mrz": primary_jobs} if primary_jobs else {}
        )
        localization.update(primary)

        def has_mrz(item_id: str) -> bool:
            result = localization.get((item_id, "mrz"))
            return result is not None and result.polygon.size == 8

        fallback_items = [
            item for item in items
            if item.mrz_fallback_for and not has_mrz(item.mrz_fallback_for)
        ]
        fallback, _fallback_errors, fallback_diagnostics = self._localize(
            {"mrz": [(item.item_id, item.image) for item in fallback_items]}
            if fallback_items else {}
        )
        localization.update(fallback)
        localization_diagnostics = self._merge_localization_diagnostics(
            localization_diagnostics, primary_diagnostics, fallback_diagnostics
        )
        fallback_by_primary = {item.mrz_fallback_for: item for item in fallback_items}
        probe_samples = {}
        for primary_id, _ in primary_jobs:
            fallback_item = fallback_by_primary.get(primary_id)
            probe_samples[primary_id] = {
                "back_scanned": True,
                "front_fallback_scanned": fallback_item is not None,
                "side_selected": "back" if has_mrz(primary_id) else "front" if fallback_item and has_mrz(fallback_item.item_id) else None,
            }
        mrz_probe_diagnostics = {
            "back_scanned": len(primary_jobs),
            "front_fallback_scanned": len(fallback_items),
            "samples": probe_samples,
        }
        prepared: dict[str, Any] = {}
        mrz_polygons: dict[str, np.ndarray] = {}
        preparation_started = time.perf_counter()
        for item in items:
            if item.item_id in errors:
                continue
            try:
                if item.localization_kind == "mrz":
                    mrz = localization[(item.item_id, "mrz")].polygon.reshape(4, 2)
                    if item.passport_page_corners is None:
                        raise ValueError("MRZ localization requires passport page geometry")
                    corners = page_corners_from_mrz_width(mrz, item.passport_page_corners)
                    mrz_polygons[item.item_id] = mrz
                else:
                    padded_corners = localization[(item.item_id, "docaligner")].polygon.reshape(4, 2)
                    corners = padded_corners - item.padding
                    if item.probe_mrz or item.mrz_fallback_for:
                        probe = localization.get((item.item_id, "mrz"))
                        if probe is not None and probe.polygon.size == 8:
                            polygon = probe.polygon
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
            OcrSample(
                f"visible:{item.item_id}",
                prepared[item.item_id].data_crop,
                prepared[item.item_id].profile.field_rois,
                "visible",
            )
            for item in items
            if item.item_id in prepared
        ]
        mrz_crops: list[tuple[str, np.ndarray]] = []
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
                if self.mrz_recognizer is None:
                    ocr_samples.append(OcrSample(f"mrz:{item.item_id}", processed, role="mrz"))
                else:
                    mrz_crops.append((item.item_id, crop))
            except (cv2.error, TypeError, ValueError) as error:
                errors[item.item_id] = error
        mrz_crop_seconds = time.perf_counter() - mrz_crop_started

        if self.ocr_grouping == "split":
            grouped_samples = {
                role: [sample for sample in ocr_samples if sample.role == role]
                for role in ("visible", "mrz")
            }
            ocr_result = self._merge_ocr_results(
                [self.ocr.run(samples) for samples in grouped_samples.values() if samples]
            )
        else:
            ocr_result = self.ocr.run(ocr_samples)
            ocr_result.diagnostics["ocr_grouping"] = "combined"
        mrz_diagnostics = {
            "backend": "generic-paddle" if self.mrz_recognizer is None else "mrzscanner",
            "configured_batch_size": self.mrz_recognition_batch_size,
            "calls": [],
            "retry_split_count": 0,
            "isolated_failure_count": 0,
        }
        mrz_results: dict[str, MrzRecognitionResult] = {}
        mrz_errors: dict[str, Exception] = {}
        mrz_started = time.perf_counter()
        if self.mrz_recognizer is not None:
            for chunk in _chunks(mrz_crops, self.mrz_recognition_batch_size):
                results, failures = self._recognize_mrz_chunk(
                    list(chunk), mrz_diagnostics["calls"], mrz_diagnostics
                )
                mrz_results.update(results)
                mrz_errors.update(failures)
        mrz_diagnostics["elapsed_wall_seconds"] = time.perf_counter() - mrz_started
        mrz_diagnostics["failure_count"] = len(mrz_errors)
        _finish_stage(mrz_diagnostics)
        errors.update(mrz_errors)
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
                    if self.mrz_recognizer is None:
                        tokens = ocr_result.tokens.get(sample_id, [])
                        selected = select(reconstruct(tokens), item.mrz_profile.line_counts)
                        mrz_text = "\n".join(line.text for line in selected)
                    else:
                        recognized = mrz_results.get(item.item_id)
                        mrz_text = "\n".join(recognized.lines) if recognized else ""
                        item.artifacts.save_json("mrz_recognition.json", {
                            "status": recognized.status if recognized else "not_found",
                            "raw_lines": list(recognized.lines) if recognized else [],
                            "score": recognized.score if recognized else None,
                        })
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
        mrz_output_signatures = {}
        for item in items:
            tokens = ocr_result.tokens.get(f"mrz:{item.item_id}", [])
            reconstructed = reconstruct(tokens)
            selected = select(reconstructed, item.mrz_profile.line_counts) if item.mrz_profile else []
            mrz_output_signatures[item.item_id] = {
                "detected_line_count": len(reconstructed),
                "detected_lines_sha256": _text_digest([line.text for line in reconstructed]),
                "reconstructed_line_count": len(selected),
                "reconstructed_lines_sha256": _text_digest([line.text for line in selected]),
            }

        diagnostics = {
            "total_wall_seconds": time.perf_counter() - started_total,
            "localization": localization_diagnostics,
            "id_card_mrz_probe": mrz_probe_diagnostics,
            "pipeline": {
                "document_preparation_seconds": preparation_seconds,
                "canonicalization_seconds": sum(value.timings.get("canonicalization_seconds", 0.0) for value in prepared.values()),
                "data_crop_seconds": sum(value.timings.get("data_crop_seconds", 0.0) for value in prepared.values()),
                "mrz_crop_preprocess_seconds": mrz_crop_seconds,
                "result_assembly_seconds": result_assembly_seconds,
            },
            **ocr_result.diagnostics,
            "mrz_recognition": mrz_diagnostics,
            "mrz_output_signatures": mrz_output_signatures,
        }
        return [outcomes_by_id[item.item_id] for item in items], diagnostics
