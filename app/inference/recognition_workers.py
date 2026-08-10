"""Process-isolated Paddle text-recognition workers."""

from __future__ import annotations

import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Sequence

import numpy as np

_recognizer: Any | None = None


def _init_recognizer(model_dir: str | None, cpu_threads: int) -> None:
    global _recognizer
    from paddleocr import TextRecognition

    _recognizer = TextRecognition(
        model_name="PP-OCRv6_medium_rec",
        model_dir=model_dir,
        device="cpu",
        cpu_threads=cpu_threads,
    )


def _predict(images: list[np.ndarray]) -> tuple[list[dict[str, Any]], float]:
    if _recognizer is None:
        raise RuntimeError("text recognition worker was not initialized")
    started = time.perf_counter()
    values = list(_recognizer.predict(input=images, batch_size=len(images)))
    if len(values) != len(images):
        raise ValueError(f"text recognition returned {len(values)} results for {len(images)} inputs")
    return [
        {
            "rec_text": str(value.get("rec_text", "") if hasattr(value, "get") else value["rec_text"]),
            "rec_score": float(value.get("rec_score", 0.0) if hasattr(value, "get") else value["rec_score"]),
        }
        for value in values
    ], time.perf_counter() - started


def _ready() -> int:
    return multiprocessing.current_process().pid


class ProcessTextRecognizer:
    """One Paddle model per spawned process; never share a predictor between threads."""

    def __init__(self, *, model_dir: str | None, processes: int, cpu_threads: int):
        self.model_dir = model_dir
        self.processes = processes
        self.cpu_threads = max(1, cpu_threads // processes)
        self._executor: ProcessPoolExecutor | None = None

    def _pool(self) -> ProcessPoolExecutor:
        if self._executor is None:
            self._executor = ProcessPoolExecutor(
                max_workers=self.processes,
                mp_context=multiprocessing.get_context("spawn"),
                initializer=_init_recognizer,
                initargs=(self.model_dir, self.cpu_threads),
            )
        return self._executor

    def predict_chunks(
        self, chunks: Sequence[list[np.ndarray]]
    ) -> list[tuple[list[dict[str, Any]], float]]:
        futures = [self._pool().submit(_predict, chunk) for chunk in chunks]
        try:
            return [future.result() for future in futures]
        except BaseException:
            self.close(wait=False)
            raise

    def start(self) -> None:
        futures = [self._pool().submit(_ready) for _ in range(self.processes)]
        for future in futures:
            future.result()

    def close(self, *, wait: bool = True) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=wait, cancel_futures=True)
            self._executor = None
