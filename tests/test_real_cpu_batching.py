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
            (models.text_detector(), detection_image, "dt_polys"),
            (models.text_recognizer(), recognition_image, "rec_text"),
        )
        for model, image, key in cases:
            predictor = model.paddlex_predictor
            spy = RunnerSpy(predictor.runner)
            predictor.runner = spy
            batched = model.predict(input=[image] * 10, batch_size=4)
            single = model.predict(input=[image], batch_size=1)[0]
            self.assertEqual(10, len(batched))
            self.assertEqual([4, 4, 2], spy.batch_sizes[:3])
            if key == "dt_polys":
                np.testing.assert_allclose(batched[0][key], single[key], atol=1)
                np.testing.assert_allclose(batched[-1][key], single[key], atol=1)
            else:
                self.assertEqual(single[key], batched[0][key])
                self.assertEqual(single[key], batched[-1][key])


if __name__ == "__main__":
    unittest.main()
