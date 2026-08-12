import os
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from app.config import Settings
from app.models import Models


MODEL_ROOT = Path.home() / ".paddlex"
HAS_MODELS = all(
    (MODEL_ROOT / "official_models" / name / "inference.pdiparams").is_file()
    for name in ("PP-OCRv6_medium_det", "PP-OCRv6_medium_rec")
)


class RunnerSpy:
    def __init__(self, runner):
        self.runner = runner
        self.batch_sizes = []

    def __call__(self, **inputs):
        tensors = inputs["x"]
        tensor = tensors[0] if isinstance(tensors, list) else tensors
        self.batch_sizes.append(int(tensor.shape[0]))
        return self.runner(**inputs)


@unittest.skipUnless(HAS_MODELS, "cached Paddle models are unavailable; no download is allowed")
class RealCpuPaddleBatchingTests(unittest.TestCase):
    def test_detection_and_recognition_execute_real_cpu_batches(self):
        with patch.dict(
            os.environ,
            {
                "RUNTIME_TARGET": "cpu",
                "OCR_DEVICE": "cpu",
                "MODEL_DIR": str(MODEL_ROOT),
                "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "true",
            },
            clear=True,
        ):
            models = Models(Settings.from_env())

        source = cv2.imread("annotation_input/passports/passport.png")
        self.assertIsNotNone(source)
        detection_image = cv2.resize(source, (320, 240))
        recognition_image = cv2.resize(source[100:220, 100:500], (240, 48))
        cases = (
            (models.text_detector(), detection_image),
            (models.text_recognizer(), recognition_image),
        )
        for adapter, image in cases:
            predictor = adapter.model.paddlex_predictor
            spy = RunnerSpy(predictor.runner)
            predictor.runner = spy
            method = adapter.detect_batch if hasattr(adapter, "detect_batch") else adapter.recognize_batch
            batched = method([image] * 2)
            single = method([image])[0]
            self.assertEqual(2, len(batched))
            self.assertEqual(2, spy.batch_sizes[0])
            if hasattr(single, "regions"):
                np.testing.assert_allclose(batched[0].regions[0].polygon, single.regions[0].polygon, atol=1)
                np.testing.assert_allclose(batched[-1].regions[0].polygon, single.regions[0].polygon, atol=1)
            else:
                self.assertEqual(single.text, batched[0].text)
                self.assertEqual(single.text, batched[-1].text)


if __name__ == "__main__":
    unittest.main()
