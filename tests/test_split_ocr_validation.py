import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np

from app.artifacts import ArtifactWriter
from app.config import Settings
from app.documents.mrz import MrzProfile
from app.inference.batch import BatchedOcr, ProfileBatchItem, ProfileBatchRunner
from app.inference.contracts import DetectedTextRegion, DetectedTextRegions, RecognitionResult
from app.inference.packing import SequentialBatchPacker
from app.pipeline import RegionProfile
from benchmarks.maintained.split_ocr_validation import (
    MODES,
    comparison_modes,
    execution_equivalence,
    measurement_stats,
    output_equivalence,
)
from benchmarks.maintained.pipeline_breakdown import Document, Run


class FakeLocalizer:
    def localize_batch(self, images):
        self.last_tensor_batch_size = len(images)
        return [type("Location", (), {"polygon": np.float32([[0, 0], [image.shape[1] - 1, 0], [image.shape[1] - 1, image.shape[0] - 1], [0, image.shape[0] - 1]]).reshape(-1)})() for image in images]


class FakeDetector:
    def __init__(self):
        self.calls = []

    def detect_batch(self, images):
        self.calls.append([image.shape[:2] for image in images])
        return [
            DetectedTextRegions((DetectedTextRegion(np.float32([[1, 1], [image.shape[1] - 2, 1], [image.shape[1] - 2, image.shape[0] - 2], [1, image.shape[0] - 2]])),))
            for image in images
        ]


class FakeRecognizer:
    def __init__(self):
        self.calls = []

    def recognize_batch(self, images):
        self.calls.append(len(images))
        return [RecognitionResult("A" * 30, 0.9) for _ in images]


def make_runner(grouping, detector, recognizer):
    settings = Settings.from_env()
    ocr = BatchedOcr(detector, recognizer, detection_batch_size=8, recognition_batch_size=8, recognition_packer=SequentialBatchPacker())
    return ProfileBatchRunner(ocr, {"docaligner": FakeLocalizer(), "mrz": FakeLocalizer()}, settings.mrz, localization_batch_size=8, max_items=4, ocr_grouping=grouping)


class SplitOcrValidationTests(unittest.TestCase):
    def test_all_groupings_execute_visible_and_mrz_samples(self):
        image = np.zeros((48, 120, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory:
            item = ProfileBatchItem(
                "card:back", image, RegionProfile({"x1": 0, "y1": 0, "x2": 1, "y2": 1}, {"all": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}}),
                "docaligner", lambda _: ({}, {}), lambda _: [], ArtifactWriter(Path(directory), "", "test", False),
                120, 48, mrz_profile=MrzProfile((1,)), probe_mrz=True,
            )
            for grouping in ("combined", "split"):
                detector, recognizer = FakeDetector(), FakeRecognizer()
                outcomes, diagnostics = make_runner(grouping, detector, recognizer).run([item])
                self.assertIsNone(outcomes[0].error)
                self.assertEqual(diagnostics["sample_counts"], {"visible": 1, "mrz": 1})
                self.assertEqual(diagnostics["line_counts_by_role"]["mrz"]["recognized"], 1)
                roles = {call["role"] for call in diagnostics["text_recognition"]["calls"]}
                self.assertEqual(roles, {"mixed"} if grouping == "combined" else {"visible", "mrz"})

    def test_crop_hashes_and_shape_padding_are_recorded(self):
        detector = FakeDetector(); recognizer = FakeRecognizer()
        engine = BatchedOcr(detector, recognizer, detection_batch_size=8, recognition_batch_size=8)
        samples = [
            __import__("app.inference.batch", fromlist=["OcrSample"]).OcrSample("visible:a", np.zeros((20, 100, 3), dtype=np.uint8), role="visible"),
            __import__("app.inference.batch", fromlist=["OcrSample"]).OcrSample("mrz:a", np.zeros((40, 40, 3), dtype=np.uint8), role="mrz"),
        ]
        diagnostics = engine.run(samples).diagnostics
        self.assertEqual([row["crop_sha256"] for row in diagnostics["sample_records"]], [row["crop_sha256"] for row in diagnostics["sample_records"]])
        call = diagnostics["text_detection"]["calls"][0]
        self.assertEqual(call["role"], "mixed")
        self.assertLess(call["padding_efficiency"], 1.0)
        self.assertEqual(call["actual_batch_size"], 2)

    def test_invalid_execution_is_excluded_and_outputs_detect_differences(self):
        base = {"sample_counts": {"visible": 1, "mrz": 1}, "line_counts_by_role": {"visible": {"recognized": 1}, "mrz": {"recognized": 1}}, "text_detection": {"by_role": {}}, "text_recognition": {"by_role": {}}, "sample_records": []}
        runs = {mode: Run(mode, "passport", 1, 1, 1, ("p",), "ok", 1.0, None, None, {}, deepcopy(base), {"p": {"final_digest": mode}}) for mode in MODES}
        runs["SPLIT_BATCH_8"].diagnostics["sample_counts"]["mrz"] = 0
        execution = execution_equivalence(runs)
        self.assertNotIn("SPLIT_BATCH_8", comparison_modes(execution))
        self.assertEqual(output_equivalence(runs)["transition_counts"]["COMBINED_BATCH_8"]["meaningfully_different"], 1)

    def test_median_aggregation_reports_mad_and_iqr(self):
        self.assertEqual(measurement_stats([1.0, 2.0, 100.0])["median"], 2.0)
        self.assertEqual(measurement_stats([1.0, 2.0, 100.0])["mad"], 1.0)


if __name__ == "__main__":
    unittest.main()
