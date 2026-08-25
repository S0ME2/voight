"""Paddle-specific text detection and recognition adapters."""

from __future__ import annotations

from collections.abc import Sequence
import math
import time
from typing import Any

import cv2
import numpy as np

from app.config import RuntimeSettings
from app.inference.contracts import (
    DetectedTextRegion,
    DetectedTextRegions,
    RecognitionResult,
)


def _value(result: Any, key: str, default: Any = None) -> Any:
    try:
        return result[key]
    except (KeyError, TypeError):
        return getattr(result, key, default)


class PaddleTextDetector:
    def __init__(self, model: Any, runtime: RuntimeSettings):
        self.model = model
        self.preserves_source_shapes = hasattr(model, "paddlex_predictor")
        self.limit_side_len_override = runtime.text_detector_limit_side_len
        self.pixel_scale = runtime.text_detector_pixel_scale

    @staticmethod
    def _shape_bucket(height: int, width: int, step: int = 1) -> tuple[int, int]:
        return max(step, math.ceil(height / step) * step), max(step, math.ceil(width / step) * step)

    def _fixed_shape_predict(self, images: Sequence[np.ndarray]) -> list[dict[str, Any]]:
        predictor = self.model.paddlex_predictor
        resize = predictor.pre_tfs["Resize"]
        limit_side_len = self.limit_side_len_override or predictor.limit_side_len
        limit_type = predictor.limit_type
        raw = predictor.pre_tfs["Read"](imgs=list(images))
        resized, shapes = predictor.pre_tfs["Resize"](
            imgs=raw,
            limit_side_len=limit_side_len,
            limit_type=limit_type,
            max_side_limit=predictor.max_side_limit,
        )
        if self.pixel_scale != 1.0:
            scaled = []
            scaled_shapes = []
            for image, shape in zip(resized, shapes):
                height = max(32, int(image.shape[0] * self.pixel_scale / 32 + 0.5) * 32)
                width = max(32, int(image.shape[1] * self.pixel_scale / 32 + 0.5) * 32)
                scaled.append(cv2.resize(image, (width, height)))
                adjusted = np.asarray(shape, dtype=np.float32).copy()
                adjusted[2] *= height / image.shape[0]
                adjusted[3] *= width / image.shape[1]
                scaled_shapes.append(adjusted)
            resized, shapes = scaled, scaled_shapes
        normalized = predictor.pre_tfs["Normalize"](imgs=resized)
        chw = predictor.pre_tfs["ToCHW"](imgs=normalized)
        groups: dict[tuple[int, int], list[tuple[int, np.ndarray, list[float]]]] = {}
        for index, (image, shape) in enumerate(zip(chw, shapes)):
            groups.setdefault(self._shape_bucket(image.shape[1], image.shape[2]), []).append((index, image, shape))
        outputs: list[dict[str, Any] | None] = [None] * len(images)
        self.last_tensor_batch_sizes = []
        self.last_tensor_shapes = []
        self.last_tensor_pixel_counts = []
        self.last_resized_shapes = [list(map(int, image.shape[:2])) for image in resized]
        self.last_resize_config = {
            "transform": type(resize).__name__,
            "predictor_limit_side_len": predictor.limit_side_len,
            "transform_limit_side_len": getattr(resize, "limit_side_len", None),
            "effective_limit_side_len": limit_side_len or getattr(resize, "limit_side_len", None),
            "limit_type": limit_type or getattr(resize, "limit_type", None),
            "max_side_limit": predictor.max_side_limit,
            "keep_ratio": getattr(resize, "keep_ratio", None),
            "rounding": "nearest multiple of 32",
            "pixel_scale": self.pixel_scale,
            "requested_pixel_fraction": self.pixel_scale ** 2,
            "pixel_scale_stage": "after DetResizeForTest, before detector tensor",
        }
        started = time.perf_counter()
        model_seconds = 0.0
        for bucket, group in sorted(groups.items()):
            batch = np.zeros((len(group), image.shape[0], bucket[0], bucket[1]), dtype=np.float32)
            batch_shapes = []
            for offset, (_, image, shape) in enumerate(group):
                batch[offset, :, : image.shape[1], : image.shape[2]] = image
                batch_shapes.append(shape)
            model_started = time.perf_counter()
            predictions = predictor.runner(x=[batch])
            model_seconds += time.perf_counter() - model_started
            polys, scores = predictor.post_op(
                predictions,
                batch_shapes,
                thresh=predictor.thresh,
                box_thresh=predictor.box_thresh,
                unclip_ratio=predictor.unclip_ratio,
            )
            self.last_tensor_batch_sizes.append(len(group))
            self.last_tensor_shapes.append([len(group), int(batch.shape[1]), *map(int, bucket)])
            self.last_tensor_pixel_counts.append(len(group) * int(bucket[0]) * int(bucket[1]))
            for (_, _, _), polygons, values, index in zip(group, polys, scores, (entry[0] for entry in group)):
                outputs[index] = {"dt_polys": polygons, "dt_scores": values}
        self.last_tensor_batch_size = max(self.last_tensor_batch_sizes, default=0)
        self.last_model_seconds = model_seconds or (time.perf_counter() - started)
        return [value or {"dt_polys": (), "dt_scores": ()} for value in outputs]

    def resize_configuration(self) -> dict[str, Any]:
        if not self.preserves_source_shapes:
            return {"backend_transform": "unavailable"}
        resize = self.model.paddlex_predictor.pre_tfs["Resize"]
        return {
            "transform": type(resize).__name__,
            "predictor_limit_side_len": self.model.paddlex_predictor.limit_side_len,
            "transform_limit_side_len": getattr(resize, "limit_side_len", None),
            "effective_limit_side_len": self.limit_side_len_override or getattr(resize, "limit_side_len", None),
            "limit_type": self.model.paddlex_predictor.limit_type or getattr(resize, "limit_type", None),
            "max_side_limit": self.model.paddlex_predictor.max_side_limit,
            "keep_ratio": getattr(resize, "keep_ratio", None),
            "rounding": "nearest multiple of 32",
            "pixel_scale": self.pixel_scale,
            "requested_pixel_fraction": self.pixel_scale ** 2,
            "pixel_scale_stage": "after DetResizeForTest, before detector tensor",
        }

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[DetectedTextRegions]:
        values = (
            self._fixed_shape_predict(images)
            if self.preserves_source_shapes
            else list(self.model.predict(input=list(images), batch_size=len(images)))
        )
        results = []
        for value in values:
            polygons = _value(value, "dt_polys", ())
            scores = _value(value, "dt_scores", ())
            polygons = () if polygons is None else polygons
            scores = () if scores is None else scores
            results.append(
                DetectedTextRegions(
                    tuple(
                        DetectedTextRegion(
                            np.asarray(polygon, dtype=np.float32),
                            float(scores[index]) if index < len(scores) else None,
                        )
                        for index, polygon in enumerate(polygons)
                    ),
                    str(error) if (error := _value(value, "error")) else None,
                )
            )
        return results


class PaddleTextRecognizer:
    def __init__(self, model: Any):
        self.model = model
        self.model_name = getattr(model, "_model_name", None)
        self.last_crop_traces: list[dict[str, int]] = []
        predictor = getattr(model, "paddlex_predictor", None)
        if predictor is not None:
            resize = predictor.pre_tfs.get("ReisizeNorm")
            to_batch = predictor.pre_tfs.get("ToBatch")
            if resize is not None and to_batch is not None:
                def traced_resize(*args, **kwargs):
                    values = resize(*args, **kwargs)
                    self._resized_shapes = [(int(value.shape[1]), int(value.shape[2])) for value in values]
                    return values

                def traced_batch(*args, **kwargs):
                    images = args[0] if args else kwargs["imgs"]
                    self._tensor_shape = (int(images[0].shape[1]), int(max(value.shape[2] for value in images)))
                    return to_batch(*args, **kwargs)

                predictor.pre_tfs["ReisizeNorm"] = traced_resize
                predictor.pre_tfs["ToBatch"] = traced_batch

    def recognize_batch(self, images: Sequence[np.ndarray]) -> list[RecognitionResult]:
        self._resized_shapes = []
        self._tensor_shape = None
        started = time.perf_counter()
        values = [
            RecognitionResult(
                str(_value(value, "rec_text", "")).strip(),
                float(score) if (score := _value(value, "rec_score")) is not None else None,
            )
            for value in self.model.predict(input=list(images), batch_size=len(images))
        ]
        tensor_h, tensor_w = self._tensor_shape or (48, max(320, *(image.shape[1] for image in images)))
        self.last_crop_traces = [
            {
                "resized_h": height,
                "resized_w": width,
                "tensor_h": tensor_h,
                "tensor_w": tensor_w,
                "useful_pixels": height * width,
                "padded_pixels": tensor_h * tensor_w,
            }
            for height, width in self._resized_shapes
        ]
        self.last_tensor_batch_size = len(images)
        self.last_tensor_batch_sizes = [len(images)]
        self.last_model_seconds = time.perf_counter() - started
        return values


class ProcessTextRecognizer:
    def __init__(self, worker: Any):
        self.worker = worker
        self.model_name = getattr(worker, "model_name", None)
        self.last_model_seconds: float | None = None

    def recognize_batch(self, images: Sequence[np.ndarray]) -> list[RecognitionResult]:
        predictions = self.worker.predict_chunks([list(images)])
        if len(predictions) != 1:
            raise ValueError("text recognition returned an unexpected number of batches")
        values, self.last_model_seconds = predictions[0]
        return [
            RecognitionResult(str(value["rec_text"]).strip(), float(value["rec_score"]))
            for value in values
        ]

    def start(self) -> None:
        self.worker.start()

    def close(self) -> None:
        self.worker.close()
