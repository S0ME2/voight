"""Recognition crop packing; no OCR or document semantics belong here."""

from __future__ import annotations

from collections.abc import Sequence

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


def recognition_batch_packer(name: str):
    if name == "sequential":
        return SequentialBatchPacker()
    if name == "aspect-ratio":
        return AspectRatioBatchPacker()
    raise ValueError(f"unknown recognition packing strategy: {name}")
