import os
import unittest
from importlib.util import find_spec
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.config import Settings
from app.inference.localization import DocAlignerBatchLocalizer, MrzScannerBatchLocalizer
from app.models import Models

HAS_LOCALIZERS = find_spec("docaligner") is not None and find_spec("mrzscanner") is not None


class EngineSpy:
    def __init__(self, engine):
        self.engine = engine
        self.input_infos = engine.input_infos
        self.output_infos = engine.output_infos
        self.providers = engine.providers
        self.batch_sizes = []

    def __call__(self, **inputs):
        self.batch_sizes.append(next(iter(inputs.values())).shape[0])
        return self.engine(**inputs)


class LocalizationBatchTests(unittest.TestCase):
    def test_project_adapters_construct_one_n_tensor_and_restore_order(self):
        class Engine:
            input_infos = {"input": {}}
            output_infos = {"output": {}}
            providers = ["CPUExecutionProvider"]

            def __init__(self, channels):
                self.channels = channels
                self.calls = []

            def __call__(self, **inputs):
                tensor = inputs["input"]
                self.calls.append(tensor.shape)
                if self.channels == 4:
                    return {"output": np.repeat(tensor[:, :1], 4, axis=1)}
                return {"output": tensor[:, 0]}

        doc_engine = Engine(4)
        doc_inference = type("DocInference", (), {"model": doc_engine, "img_size_infer": (2, 2)})()
        aligner = type("Aligner", (), {"detector": doc_inference})()

        def doc_preprocess(*, img, **_):
            value = float(img[0, 0, 0])
            return {"input": {"img": np.full((1, 3, 2, 2), value, np.float32)}, "img_size_ori": img.shape[:2]}

        def doc_postprocess(*, preds, **_):
            return np.full((4, 2), preds[0, 0, 0, 0], np.float32)

        doc = DocAlignerBatchLocalizer(aligner, preprocess=doc_preprocess, postprocess=doc_postprocess)
        images = [np.full((4, 4, 3), marker, np.uint8) for marker in (1, 2, 3)]
        output = doc.predict_batch(images)
        self.assertEqual([(3, 3, 2, 2)], doc_engine.calls)
        self.assertEqual([1, 2, 3], [int(item["corners"][0, 0]) for item in output])

        mrz_engine = Engine(1)

        class MrzInference:
            model = mrz_engine

            def preprocess(self, image, normalize=True):
                value = float(image[0, 0, 0])
                return {"input": np.full((1, 3, 2, 2), value, np.float32)}, image.shape[:2], (0, 0)

            def postprocess(self, *, hmap, **_):
                return np.full((4, 2), hmap[0, 0], np.float32)

        mrz = MrzScannerBatchLocalizer(type("Scanner", (), {"detector": MrzInference()})())
        output = mrz.predict_batch(images)
        self.assertEqual([(3, 3, 2, 2)], mrz_engine.calls)
        self.assertEqual([1, 2, 3], [int(item["mrz_polygon"][0, 0]) for item in output])

    @unittest.skipUnless(HAS_LOCALIZERS, "CPU localization wrappers are unavailable")
    def test_real_cpu_onnx_sessions_accept_n_greater_than_one(self):
        with patch.dict(os.environ, {"RUNTIME_TARGET": "cpu", "OCR_DEVICE": "cpu"}, clear=True):
            models = Models(Settings.from_env())
        images = [np.zeros((64, 96, 3), np.uint8), np.full((80, 60, 3), 20, np.uint8)]
        for adapter in (models.document_localizer(), models.mrz_localizer()):
            self.assertEqual(["CPUExecutionProvider"], adapter.providers)
            spy = EngineSpy(adapter.engine)
            adapter.engine = spy
            batched = adapter.predict_batch(images)
            singles = [adapter.predict_batch([image])[0] for image in images]
            self.assertEqual(2, spy.batch_sizes[0])
            key = "corners" if "corners" in batched[0] else "mrz_polygon"
            for actual, expected in zip(batched, singles):
                np.testing.assert_allclose(actual[key], expected[key], atol=1e-4)


if __name__ == "__main__":
    unittest.main()
