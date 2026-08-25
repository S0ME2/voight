import unittest

import numpy as np

from app.inference.batch import BatchedOcr, OcrSample, _pad_detection_batch
from app.inference.contracts import DetectedTextRegion, DetectedTextRegions, RecognitionResult
from app.inference.packing import AspectRatioBatchPacker, FixedWidthBatchPacker


class BatchPackingTests(unittest.TestCase):
    def test_aspect_ratio_order_is_stable_and_results_restore(self):
        widths = {10: 12, 20: 52, 30: 22, 40: 42}

        class Detector:
            def detect_batch(self, images):
                return [
                    DetectedTextRegions((DetectedTextRegion(np.asarray(
                        [[2, 2], [widths[int(image[0, 0, 0])], 2], [widths[int(image[0, 0, 0])], 7], [2, 7]],
                        np.float32,
                    )),))
                    for image in images
                ]

        class Recognizer:
            def __init__(self):
                self.ratios = []

            def recognize_batch(self, images):
                self.ratios.extend(image.shape[1] / image.shape[0] for image in images)
                return [RecognitionResult(str(int(round(float(image.mean())))), 1.0) for image in images]

        recognizer = Recognizer()
        result = BatchedOcr(
            Detector(), recognizer,
            detection_batch_size=4,
            recognition_batch_size=4,
            recognition_packer=AspectRatioBatchPacker(),
        ).run([
            OcrSample(str(marker), np.full((20, 60, 3), marker, np.uint8))
            for marker in (10, 20, 30, 40)
        ])
        self.assertEqual(sorted(recognizer.ratios), recognizer.ratios)
        self.assertEqual(["10", "20", "30", "40"], [result.tokens[str(marker)][0]["text"] for marker in (10, 20, 30, 40)])
        self.assertEqual("aspect-ratio", result.diagnostics["text_recognition"]["packing_strategy"])
        self.assertTrue(result.diagnostics["text_recognition"]["original_index_restoration_success"])
        self.assertGreater(result.diagnostics["text_recognition"]["calls"][0]["padding_efficiency_estimate"], 0)

    def test_detection_padding_is_retained_for_varied_shapes(self):
        padded = _pad_detection_batch([
            (0, np.zeros((20, 40, 3), np.uint8)),
            (1, np.zeros((30, 10, 3), np.uint8)),
        ])
        self.assertEqual([(30, 40, 3), (30, 40, 3)], [image.shape for _, image in padded])
        self.assertEqual([0, 1], [index for index, _ in padded])

    def test_fixed_width_keeps_crop_tensor_independent_of_batch_members(self):
        packer = FixedWidthBatchPacker()
        image = np.full((20, 60, 3), 80, np.uint8)
        unrelated = np.full((20, 60, 3), 160, np.uint8)
        solo = packer.pack([(0, image)], 1)[0][0][1]
        pair = packer.pack([(0, image), (1, unrelated)], 2)[0]
        self.assertEqual(solo.shape, pair[0][1].shape)
        np.testing.assert_array_equal(solo, pair[0][1])


if __name__ == "__main__":
    unittest.main()
