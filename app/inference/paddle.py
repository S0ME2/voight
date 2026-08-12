"""Paddle-specific text detection and recognition adapters."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

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
    def __init__(self, model: Any):
        self.model = model

    def detect_batch(self, images: Sequence[np.ndarray]) -> list[DetectedTextRegions]:
        values = list(self.model.predict(input=list(images), batch_size=len(images)))
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

    def recognize_batch(self, images: Sequence[np.ndarray]) -> list[RecognitionResult]:
        return [
            RecognitionResult(
                str(_value(value, "rec_text", "")).strip(),
                float(score) if (score := _value(value, "rec_score")) is not None else None,
            )
            for value in self.model.predict(input=list(images), batch_size=len(images))
        ]


class ProcessTextRecognizer:
    def __init__(self, worker: Any):
        self.worker = worker
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
