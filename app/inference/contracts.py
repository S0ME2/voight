"""Project-owned model boundaries used by inference orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class LocalizationResult:
    polygon: np.ndarray
    score: float | None = None


@dataclass(frozen=True)
class DetectedTextRegion:
    polygon: np.ndarray
    score: float | None = None


@dataclass(frozen=True)
class DetectedTextRegions:
    regions: tuple[DetectedTextRegion, ...]
    error: str | None = None


@dataclass(frozen=True)
class RecognitionResult:
    text: str
    score: float | None = None


@dataclass(frozen=True)
class MrzRecognitionResult:
    lines: tuple[str, ...]
    status: str
    score: float | None = None


class DocumentLocalizer(Protocol):
    def localize_batch(self, images: Sequence[np.ndarray]) -> list[LocalizationResult]: ...


class TextDetector(Protocol):
    def detect_batch(self, images: Sequence[np.ndarray]) -> list[DetectedTextRegions]: ...


class TextRecognizer(Protocol):
    def recognize_batch(self, images: Sequence[np.ndarray]) -> list[RecognitionResult]: ...


class MrzRecognizer(Protocol):
    def recognize_batch(self, images: Sequence[np.ndarray]) -> list[MrzRecognitionResult]: ...


class RecognitionBatchPacker(Protocol):
    name: str

    def pack(
        self, items: Sequence[tuple[int, np.ndarray]], batch_size: int
    ) -> list[list[tuple[int, np.ndarray]]]: ...
