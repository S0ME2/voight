import unittest
from types import ModuleType
from pathlib import Path
from unittest.mock import patch

import numpy as np

from app.artifacts import ArtifactWriter
from app.config import (
    ArtifactSettings,
    BatchSettings,
    DrivingLicenseSettings,
    MrzSettings,
    OcrSettings,
    RuntimeSettings,
    Settings,
)
from app.inference.batch import (
    BatchedOcr,
    InferenceGate,
    OcrSample,
    ProfileBatchItem,
    ProfileBatchRunner,
    QueueFullError,
)
from app.pipeline import RegionProfile
from app.models import Models


WRITER = ArtifactWriter(Path("/unused"), "", "", False)


class DetectionStub:
    def __init__(self, line_counts=None, failing_marker=None):
        self.line_counts = line_counts or {}
        self.failing_marker = failing_marker
        self.batch_sizes = []

    def predict(self, images):
        self.batch_sizes.append(len(images))
        markers = [int(image[0, 0, 0]) for image in images]
        if self.failing_marker in markers:
            raise ValueError(f"bad marker {self.failing_marker}")
        results = []
        for marker, image in zip(markers, images):
            height, width = image.shape[:2]
            count = self.line_counts.get(marker, 1)
            polygons = []
            for index in range(count):
                top = 2 + index * 8
                polygons.append(
                    [[2, top], [width - 3, top], [width - 3, top + 5], [2, top + 5]]
                )
            results.append({"dt_polys": np.asarray(polygons, dtype=np.float32)})
        return results


class RecognitionStub:
    def __init__(self, failing_marker=None):
        self.failing_marker = failing_marker
        self.batch_sizes = []

    def predict(self, images):
        self.batch_sizes.append(len(images))
        markers = [int(round(float(image.mean()))) for image in images]
        if self.failing_marker in markers:
            raise ValueError(f"bad line marker {self.failing_marker}")
        return [
            {"rec_text": f"value-{marker}", "rec_score": marker / 100}
            for marker in markers
        ]


def image(marker):
    return np.full((40, 60, 3), marker, dtype=np.uint8)


def full_document(value):
    height, width = value.shape[:2]
    return [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]


def parse(assignments):
    raw = {
        field: " ".join(token["text"] for token in tokens)
        for field, tokens in assignments.items()
    }
    return raw, raw


class BatchedOcrTests(unittest.TestCase):
    def test_true_batches_restore_variable_line_counts_and_match_single_items(self):
        detector = DetectionStub({10: 1, 20: 3, 30: 2})
        recognizer = RecognitionStub()
        engine = BatchedOcr(
            detector,
            recognizer,
            detection_batch_size=2,
            recognition_batch_size=4,
        )
        samples = [
            OcrSample("passport", image(10)),
            OcrSample("id_card", image(20)),
            OcrSample("driving_license", image(30)),
        ]
        result = engine.run(samples)

        self.assertEqual([], list(result.errors))
        self.assertEqual(
            {"passport": 1, "id_card": 3, "driving_license": 2},
            {key: len(tokens) for key, tokens in result.tokens.items()},
        )
        self.assertEqual([2, 1], detector.batch_sizes)
        self.assertEqual([4, 2], recognizer.batch_sizes)
        self.assertIn(
            2,
            result.diagnostics["text_detection"]["actual_tensor_batch_sizes"],
        )
        self.assertIn(
            4,
            result.diagnostics["text_recognition"]["actual_tensor_batch_sizes"],
        )

        for sample in samples:
            single = BatchedOcr(
                DetectionStub(detector.line_counts),
                RecognitionStub(),
                detection_batch_size=2,
                recognition_batch_size=4,
            ).run([sample])
            self.assertEqual(single.tokens[sample.item_id], result.tokens[sample.item_id])

    def test_batch_failure_splits_until_only_bad_item_fails(self):
        detector = DetectionStub(failing_marker=99)
        engine = BatchedOcr(
            detector,
            RecognitionStub(),
            detection_batch_size=3,
            recognition_batch_size=8,
        )
        result = engine.run(
            [
                OcrSample("first", image(10)),
                OcrSample("bad", image(99)),
                OcrSample("last", image(30)),
            ]
        )

        self.assertEqual(["first", "last"], list(result.tokens))
        self.assertEqual(["bad"], list(result.errors))
        self.assertGreater(result.diagnostics["text_detection"]["failure_count"], 0)
        self.assertIn(3, detector.batch_sizes)

    def test_recognition_failure_does_not_corrupt_sibling_documents(self):
        recognizer = RecognitionStub(failing_marker=99)
        engine = BatchedOcr(
            DetectionStub(),
            recognizer,
            detection_batch_size=3,
            recognition_batch_size=3,
        )
        result = engine.run(
            [
                OcrSample("first", image(10)),
                OcrSample("bad", image(99)),
                OcrSample("last", image(30)),
            ]
        )

        self.assertEqual(["first", "last"], list(result.tokens))
        self.assertEqual(["bad"], list(result.errors))
        self.assertGreater(result.diagnostics["text_recognition"]["failure_count"], 0)
        self.assertIn(3, recognizer.batch_sizes)


class ProfileBatchRunnerTests(unittest.TestCase):
    def _item(self, item_id, marker):
        profile = RegionProfile(
            {"x1": 0, "y1": 0, "x2": 1, "y2": 1},
            {"field": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}},
        )
        return ProfileBatchItem(
            item_id,
            image(marker),
            profile,
            full_document,
            parse,
            lambda _values: [],
            WRITER,
            60,
            40,
        )

    def test_all_document_regions_share_batch_path_and_keep_order(self):
        runner = ProfileBatchRunner(
            BatchedOcr(
                DetectionStub(),
                RecognitionStub(),
                detection_batch_size=8,
                recognition_batch_size=8,
            ),
            localization_batch_size=4,
            max_items=6,
            queue_limit=1,
        )
        items = [
            self._item("passport:data_page", 10),
            self._item("id_card:front", 20),
            self._item("id_card:back", 30),
            self._item("driving_license:data", 40),
        ]
        outcomes, diagnostics = runner.run(items)

        self.assertEqual(
            [item.item_id for item in items],
            [item.item_id for item in outcomes],
        )
        self.assertTrue(all(outcome.error is None for outcome in outcomes))
        self.assertEqual(
            ["value-10", "value-20", "value-30", "value-40"],
            [outcome.result[0]["field"] for outcome in outcomes],
        )
        self.assertEqual(
            [1, 1, 1, 1],
            diagnostics["localization"]["actual_tensor_batch_sizes"],
        )
        self.assertFalse(diagnostics["localization"]["batching_supported"])
        self.assertEqual(
            [4], diagnostics["text_detection"]["actual_tensor_batch_sizes"]
        )
        self.assertEqual(
            [4], diagnostics["text_recognition"]["actual_tensor_batch_sizes"]
        )

    def test_size_and_queue_limits_are_enforced(self):
        runner = ProfileBatchRunner(
            BatchedOcr(
                DetectionStub(),
                RecognitionStub(),
                detection_batch_size=1,
                recognition_batch_size=1,
            ),
            localization_batch_size=1,
            max_items=1,
            queue_limit=1,
        )
        with self.assertRaisesRegex(ValueError, "maximum is 1"):
            runner.run([self._item("a", 10), self._item("b", 20)])

        gate = InferenceGate(1)
        gate.__enter__()
        try:
            with self.assertRaises(QueueFullError):
                gate.__enter__()
        finally:
            gate.__exit__(None, None, None)


class ModelOwnerTests(unittest.TestCase):
    def test_models_owns_one_cpu_batch_runner_with_configured_sizes(self):
        created = []

        class Wrapper:
            def __init__(self, **kwargs):
                created.append(kwargs)

        paddleocr = ModuleType("paddleocr")
        paddleocr.TextDetection = Wrapper
        paddleocr.TextRecognition = Wrapper
        root = Path("/unused")
        settings = Settings(
            False,
            ArtifactSettings(False, root),
            OcrSettings("cpu"),
            MrzSettings("stub", 100, 1.0, 0.0),
            DrivingLicenseSettings(root, root, 10, 10, 0, 0.3, "stub"),
            batch=BatchSettings(max_files=3),
            runtime=RuntimeSettings(
                target="cpu",
                queue_limit=2,
                localization_batch_size=3,
                text_detection_batch_size=4,
                text_recognition_batch_size=5,
            ),
        )
        models = Models(settings)
        with patch.dict("sys.modules", {"paddleocr": paddleocr}):
            runner = models.profile_batch_runner()
            self.assertIs(runner, models.profile_batch_runner())

        self.assertEqual([4, 5], [item["batch_size"] for item in created])
        self.assertEqual(["cpu", "cpu"], [item["device"] for item in created])
        self.assertEqual(6, runner.max_items)


if __name__ == "__main__":
    unittest.main()
