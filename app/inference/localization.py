"""True-batch adapters for the upstream DocAligner and MRZScanner models."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any
import time

import numpy as np


def _engine_input_name(engine: Any) -> str:
    return next(iter(engine.input_infos))


def _engine_output_name(engine: Any) -> str:
    return next(iter(engine.output_infos))


class DocAlignerBatchLocalizer:
    """Batch the dynamic-N heatmap graph hidden by DocAligner's batch-1 wrapper."""

    def __init__(
        self,
        aligner: Any,
        *,
        preprocess: Callable[..., dict[str, Any]] | None = None,
        postprocess: Callable[..., Any] | None = None,
    ):
        inference = aligner.detector
        if preprocess is None or postprocess is None:
            from docaligner.heatmap_reg.infer import (
                postprocess as upstream_postprocess,
                preprocess as upstream_preprocess,
            )

            preprocess = preprocess or upstream_preprocess
            postprocess = postprocess or upstream_postprocess
        self.inference = inference
        self.engine = inference.model
        self.preprocess = preprocess
        self.postprocess = postprocess

    @property
    def providers(self) -> list[str]:
        return list(self.engine.providers)

    def predict_batch(self, images: Sequence[np.ndarray]) -> list[dict[str, Any]]:
        infos = [
            self.preprocess(
                img=image,
                img_size_infer=self.inference.img_size_infer,
                do_center_crop=False,
            )
            for image in images
        ]
        tensor = np.concatenate([info["input"]["img"] for info in infos], axis=0)
        self.last_tensor_batch_size = int(tensor.shape[0])
        started = time.perf_counter()
        outputs = self.engine(**{_engine_input_name(self.engine): tensor})
        self.last_model_seconds = time.perf_counter() - started
        heatmaps = outputs[_engine_output_name(self.engine)]
        if heatmaps.shape[0] != len(images):
            raise ValueError("DocAligner returned a different batch dimension")
        return [
            {
                "corners": np.asarray(
                    self.postprocess(
                        preds=heatmaps[index : index + 1],
                        imgs_size=info["img_size_ori"],
                    ),
                    dtype=np.float32,
                ),
                "tensor_batch_size": int(tensor.shape[0]),
            }
            for index, info in enumerate(infos)
        ]


class MrzScannerBatchLocalizer:
    """Batch the dynamic-N MRZ heatmap graph hidden by MRZScanner's wrapper."""

    def __init__(self, scanner: Any):
        self.inference = scanner.detector
        self.engine = self.inference.model

    @property
    def providers(self) -> list[str]:
        return list(self.engine.providers)

    def predict_batch(self, images: Sequence[np.ndarray]) -> list[dict[str, Any]]:
        infos = [self.inference.preprocess(image, normalize=True) for image in images]
        input_name = _engine_input_name(self.engine)
        tensor = np.concatenate([info[0][input_name] for info in infos], axis=0)
        self.last_tensor_batch_size = int(tensor.shape[0])
        started = time.perf_counter()
        outputs = self.engine(**{input_name: tensor})
        self.last_model_seconds = time.perf_counter() - started
        heatmaps = outputs[_engine_output_name(self.engine)]
        if heatmaps.shape[0] != len(images):
            raise ValueError("MRZScanner returned a different batch dimension")
        return [
            {
                "mrz_polygon": self.inference.postprocess(
                    hmap=heatmaps[index],
                    img_size=info[1],
                    shift=info[2],
                ),
                "tensor_batch_size": int(tensor.shape[0]),
            }
            for index, info in enumerate(infos)
        ]
