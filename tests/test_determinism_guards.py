import hashlib
import json
import unittest

import numpy as np

from app.inference.batch import BatchedOcr, OcrSample
from app.inference.contracts import DetectedTextRegion, DetectedTextRegions, RecognitionResult
from app.inference.packing import FixedWidthBatchPacker


class DeterminismGuardsTests(unittest.TestCase):
    def _run(self, detection_batch_size, recognition_batch_size):
        class Detector:
            preserves_source_shapes = True
            def detect_batch(self, images):
                return [DetectedTextRegions((DetectedTextRegion(np.asarray([[0, 0], [image.shape[1] - 1, 0], [image.shape[1] - 1, image.shape[0] - 1], [0, image.shape[0] - 1]], np.float32)),)) for image in images]

        class Recognizer:
            def recognize_batch(self, images):
                return [RecognitionResult(str(int(image[0, 0, 0])), 0.91) for image in images]

        samples = [
            OcrSample("a", np.full((20, 61, 3), 11, np.uint8)),
            OcrSample("b", np.full((30, 37, 3), 22, np.uint8)),
            OcrSample("c", np.full((18, 113, 3), 33, np.uint8)),
        ]
        return BatchedOcr(Detector(), Recognizer(), detection_batch_size=detection_batch_size, recognition_batch_size=recognition_batch_size, recognition_packer=FixedWidthBatchPacker()).run(samples)

    @staticmethod
    def _crop_records(result):
        records = [record for call in result.diagnostics["text_recognition"]["calls"] for record in call.get("crop_records", [])]
        return sorted((record["sample_id"], record["crop_sha256"], record["packed_model_input_sha256"]) for record in records)

    def test_same_request_has_one_semantic_digest_and_crop_hashes_ignore_batch_sizes(self):
        first = self._run(1, 1)
        second = self._run(3, 2)
        semantic = lambda result: hashlib.sha256(json.dumps(result.tokens, sort_keys=True, default=str).encode()).hexdigest()
        self.assertEqual(semantic(first), semantic(second))
        self.assertEqual(self._crop_records(first), self._crop_records(second))
        self.assertEqual("e6a7308c27a7d9bd19e932c6f66073a3889057b80ca8b3477e89af18818e37da", semantic(first))


if __name__ == "__main__":
    unittest.main()
