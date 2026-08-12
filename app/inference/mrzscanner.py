"""MRZScanner-specific recognition adapter."""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Any

import numpy as np

from app.inference.contracts import MrzRecognitionResult


class MrzScannerRecognizer:
    """Batch the dynamic-N recognition graph hidden by the single-image wrapper."""

    def __init__(self, scanner: Any):
        self.scanner = scanner
        self.inference = scanner.recognizer
        self.engine = self.inference.model
        info = self.engine.input_infos.get(self.inference.input_key, {})
        self.supports_batch = info.get("shape", [None])[0] != 1

    @property
    def providers(self) -> list[str]:
        return list(self.engine.providers)

    def recognize_batch(self, images: Sequence[np.ndarray]) -> list[MrzRecognitionResult]:
        tensors = [self.inference.preprocess(image, normalize=True) for image in images]
        input_name = self.inference.input_key
        output_name = self.inference.output_key
        batches = [
            np.concatenate([tensor[input_name] for tensor in tensors], axis=0)
        ] if self.supports_batch else [tensor[input_name] for tensor in tensors]
        outputs = []
        self.last_tensor_batch_sizes = []
        started = time.perf_counter()
        for batch in batches:
            outputs.append(self.engine(**{input_name: batch})[output_name])
            self.last_tensor_batch_sizes.append(int(batch.shape[0]))
        self.last_model_seconds = time.perf_counter() - started
        output = np.concatenate(outputs, axis=0)
        self.last_tensor_batch_size = max(self.last_tensor_batch_sizes)
        if output.shape[0] != len(images):
            raise ValueError("MRZScanner recognition returned a different batch dimension")
        results = []
        for index in range(len(images)):
            text = self.inference.postprocess({output_name: output[index : index + 1]})
            lines = tuple(line for line in text.split(self.inference.delimeter) if line)
            results.append(MrzRecognitionResult(lines, "recognized" if lines else "not_found"))
        return results
