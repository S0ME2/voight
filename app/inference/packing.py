"""Recognition crop packing; no OCR or document semantics belong here."""

from __future__ import annotations

from collections.abc import Sequence
import math

import cv2
import numpy as np


class SequentialBatchPacker:
    name = "sequential"

    def pack(self, items: Sequence[tuple[int, np.ndarray]], batch_size: int):
        return [list(items[start : start + batch_size]) for start in range(0, len(items), batch_size)]


class AspectRatioBatchPacker:
    name = "aspect-ratio"

    def pack(self, items: Sequence[tuple[int, np.ndarray]], batch_size: int):
        ordered = sorted(items, key=lambda item: (item[1].shape[1] / max(1, item[1].shape[0]), item[0]))
        return [ordered[start : start + batch_size] for start in range(0, len(ordered), batch_size)]


class FixedWidthBatchPacker:
    """Keep each crop's normalized width independent of its neighbours."""

    name = "fixed-width"

    def __init__(self, *, image_height: int = 48, width_step: int = 1, max_width: int = 3200):
        self.image_height = image_height
        self.width_step = width_step
        self.max_width = max_width

    def _bucket(self, image: np.ndarray) -> int:
        height, width = image.shape[:2]
        content_width = min(self.max_width, max(1, math.ceil(self.image_height * width / height)))
        resized_width = max(self.image_height * 320 // 48, content_width)
        return min(
            self.max_width,
            max(self.width_step, math.ceil(resized_width / self.width_step) * self.width_step),
        )

    def _prepare(self, image: np.ndarray, bucket_width: int) -> np.ndarray:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        height, width = image.shape[:2]
        resized_width = min(self.max_width, max(1, math.ceil(self.image_height * width / height)))
        resized = cv2.resize(image, (resized_width, self.image_height))
        canvas = np.full(
            (self.image_height, bucket_width, resized.shape[2]),
            128,
            dtype=resized.dtype,
        )
        canvas[:, :resized_width] = resized
        return canvas

    def pack(self, items: Sequence[tuple[int, np.ndarray]], batch_size: int):
        buckets: dict[int, list[tuple[int, np.ndarray]]] = {}
        for index, image in items:
            bucket = self._bucket(image)
            buckets.setdefault(bucket, []).append((index, self._prepare(image, bucket)))
        batches = []
        for bucket in sorted(buckets):
            values = buckets[bucket]
            batches.extend(values[start : start + batch_size] for start in range(0, len(values), batch_size))
        return batches


class FixedWidthBucketsBatchPacker:
    """Group untouched crops by PaddleOCR's final recognition width."""

    name = "fixed-width-buckets"

    def __init__(self, *, image_height: int = 48, minimum_width: int = 320, max_width: int = 3200):
        self.image_height = image_height
        self.minimum_width = minimum_width
        self.max_width = max_width

    def _width(self, image: np.ndarray) -> int:
        height, width = image.shape[:2]
        return min(self.max_width, max(self.minimum_width, math.ceil(self.image_height * width / height)))

    def pack(self, items: Sequence[tuple[int, np.ndarray]], batch_size: int):
        buckets: dict[int, list[tuple[int, np.ndarray]]] = {}
        for index, image in items:
            buckets.setdefault(self._width(image), []).append((index, image))
        return [
            values[start : start + batch_size]
            for width in sorted(buckets)
            for values in (buckets[width],)
            for start in range(0, len(values), batch_size)
        ]


class BestFitBatchPacker:
    """Pair adjacent final widths; for batch two this minimizes max-width pixels."""

    name = "best-fit"

    def pack(self, items: Sequence[tuple[int, np.ndarray]], batch_size: int):
        if batch_size != 2:
            raise ValueError("best-fit packing is defined for recognition max batch=2")
        ordered = sorted(items, key=lambda item: (48 * item[1].shape[1] / max(1, item[1].shape[0]), item[0]))
        return [ordered[start : start + batch_size] for start in range(0, len(ordered), batch_size)]


def recognition_batch_packer(name: str):
    if name == "sequential":
        return SequentialBatchPacker()
    if name == "aspect-ratio":
        return AspectRatioBatchPacker()
    if name == "fixed-width":
        return FixedWidthBatchPacker()
    if name == "fixed-width-buckets":
        return FixedWidthBucketsBatchPacker()
    if name == "best-fit":
        return BestFitBatchPacker()
    raise ValueError(f"unknown recognition packing strategy: {name}")
