import json
import tempfile
from unittest.mock import patch
import unittest
from pathlib import Path

import numpy as np

from benchmarks.maintained.pipeline_breakdown import (
    Document,
    Run,
    _batch_details,
    _localize,
    _stage_values,
    cycles,
    consumed_physical_count,
    score,
    stability,
    stats,
    variant_scope,
    run_partial,
)
from app.config import Settings
from app.inference.contracts import DetectedTextRegion, DetectedTextRegions, RecognitionResult
from benchmarks.historical.optimization_six import fixed_rows


class PipelineBreakdownTests(unittest.TestCase):
    def test_final_stage_report_keeps_requested_buckets_and_batch_evidence(self):
        diagnostics = {
            "localization": {"docaligner": {"wall_seconds": 1.0}},
            "pipeline": {"canonicalization_seconds": 0.2, "mrz_crop_preprocess_seconds": 0.1, "parsing_validation_seconds": 0.3},
            "text_detection": {"wall_seconds": 0.4, "calls": [{"tensor_batch_size": 2, "submitted_input_shapes": [[10, 20, 3], [10, 20, 3]]}]},
            "text_recognition": {"wall_seconds": 0.5, "calls": [{"tensor_batch_size": 2, "input_widths": [20, 30], "input_heights": [10, 10]}], "packing_strategy": "aspect-ratio"},
            "mrz_recognition": {"wall_seconds": 0.6, "calls": []},
            "line_crop_seconds": 0.05,
            "line_filter": {"detected_line_count": 3, "recognition_candidate_count": 2, "filtered_before_recognition_count": 1},
            "process_peak_rss_mb": 123.0,
        }
        stages = _stage_values(diagnostics, 3.5)
        self.assertEqual(set(stages), {"localization", "canonicalization", "detection", "roi_filtering_cropping", "recognition", "mrz_work", "parsing_validation", "other"})
        self.assertAlmostEqual(stages["mrz_work"], 0.7)
        details = _batch_details(diagnostics)
        self.assertEqual(details["detection"][0]["tensor_batch_size"], 2)
        self.assertEqual(details["recognition"][0]["input_widths"], [20, 30])
        self.assertEqual(details["process_peak_rss_mb"], 123.0)

    def test_fixed_rows_preserve_expected_order_and_count(self):
        image = np.zeros((300, 40, 3), dtype=np.uint8)
        rows = fixed_rows(image, 3)
        self.assertEqual(len(rows), 3)
        self.assertGreaterEqual(rows[0].shape[0], 100)
        self.assertGreaterEqual(rows[1].shape[0], 100)
        self.assertGreaterEqual(rows[2].shape[0], 100)
    def test_statistics_keep_median_mad_and_iqr(self):
        self.assertEqual(stats([1.0, 2.0, 100.0]), {"median": 2.0, "min": 1.0, "max": 100.0, "mad": 1.0, "iqr": 49.5})

    def test_environment_counts_unique_core_socket_pairs(self):
        from benchmarks.maintained.pipeline_breakdown import environment
        settings = Settings.from_env()
        payload = "# CPU,Core,Socket\n0,0,0\n1,0,0\n2,1,0\n3,1,0\n4,0,1\n"
        with patch("benchmarks.maintained.pipeline_breakdown.subprocess.check_output", side_effect=lambda command, **kwargs: payload if command[:2] == ["lscpu", "-p=CPU,CORE,SOCKET"] else "commit\n"):
            value = environment(settings, {"counts": {}}, "now")
        self.assertEqual(value["cpu"]["physical_cores"], 3)

    def test_scaling_order_cycles_deterministically_and_counts_id_sides(self):
        docs = [
            Document("id_card", "id_1", (("front", Path("f")), ("back", Path("b"))), Path("a")),
            Document("id_card", "id_2", (("front", Path("f2")), ("back", Path("b2"))), Path("b")),
        ]
        selected = cycles(docs, "id_card", 5)
        self.assertEqual([doc.document_id for doc in selected], ["id_1", "id_2", "id_1", "id_2", "id_1"])
        self.assertEqual(sum(doc.physical_count for doc in selected), 10)
        self.assertEqual(consumed_physical_count("id_card", "id_card_visible_known_side", selected), 10)
        self.assertEqual(consumed_physical_count("id_card", "id_card_mrz_known_back", selected), 5)

    def test_variant_scope_makes_skipped_ocr_explicit(self):
        self.assertEqual(variant_scope("passport", "passport_localization_preparation_only"), {"visible_ocr": False, "mrz_ocr": False, "preparation": True})
        self.assertEqual(variant_scope("passport", "passport_mrz_only"), {"visible_ocr": False, "mrz_ocr": True, "preparation": False})
        self.assertFalse(variant_scope("id_card", "id_card_visible_known_side")["mrz_ocr"])

    def test_localizer_records_submitted_and_actual_tensor_batches(self):
        class FakeLocalizer:
            def __init__(self):
                self.calls = []

            def localize_batch(self, images):
                self.calls.append(len(images))
                self.last_tensor_batch_size = len(images)
                return [type("Location", (), {"polygon": np.zeros((8,), dtype=np.float32)})() for _ in images]

        localizer = FakeLocalizer()
        result, diagnostics = _localize(localizer, [(str(i), np.zeros((4, 4, 3), dtype=np.uint8)) for i in range(5)], 2)
        self.assertEqual(localizer.calls, [2, 2, 1])
        self.assertEqual(diagnostics["tensor_batch_sizes"], [2, 2, 1])
        self.assertEqual(sorted(result), ["0", "1", "2", "3", "4"])

    def test_visible_and_mrz_annotations_are_scored_independently(self):
        with tempfile.TemporaryDirectory() as directory:
            annotation = Path(directory) / "p.json"
            annotation.write_text(json.dumps({
                "fields": {"known": {"state": "value", "value": "ABC"}, "empty": {"state": "empty", "value": None}, "unreadable": {"state": "unreadable", "value": None}},
                "mrz": {"lines": ["ABC", None]},
            }))
            document = Document("passport", "p", (("image", Path("image")),), annotation)
            run = Run("passport_visible_no_mrz_ocr", "passport", 1, 1, 1, ("p",), "ok", 1.0, None, None, {}, {}, {"p": {"fields": {"known": "ABC", "empty": None}, "mrz": []}})
            result = score([document], [run])
            visible = result["passport_visible_no_mrz_ocr@1:1"]["visible"]
            mrz = result["passport_visible_no_mrz_ocr@1:1"]["mrz"]
            self.assertEqual((visible["evaluated"], visible["exact"]), (2, 2))
            self.assertEqual(mrz["lines"], 1)
            self.assertEqual(mrz["line_exact"], 0)

    def test_output_stability_detects_field_and_mrz_changes(self):
        base = dict(variant="x", document_type="passport", repeat=1, logical_count=1, physical_count=1, source_ids=("p",), status="ok", total_seconds=1.0, client_seconds=None, server_seconds=None, stages={}, diagnostics={})
        first = Run(outputs={"p": {"fields": {"name": "A"}, "mrz": ["X"]}}, **base)
        second = Run(outputs={"p": {"fields": {"name": "B"}, "mrz": ["Y"]}}, **{**base, "repeat": 2})
        self.assertEqual(stability([first, second]), {"documents_with_unstable_output": 1, "fields_with_unstable_output": 1, "mrz_lines_with_unstable_output": 1})

    def test_driving_variants_call_only_their_declared_ocr_stages(self):
        class Localizer:
            def localize_batch(self, images):
                self.last_tensor_batch_size = len(images)
                return [type("Location", (), {"polygon": np.float32([[100, 100], [image.shape[1] - 100, 100], [image.shape[1] - 100, image.shape[0] - 100], [100, image.shape[0] - 100]]).reshape(-1)})() for image in images]

        class Detector:
            def __init__(self): self.calls = 0
            def detect_batch(self, images):
                self.calls += 1
                polygon = np.float32([[20, 20], [220, 20], [220, 50], [20, 50]])
                return [DetectedTextRegions((DetectedTextRegion(polygon),)) for _ in images]

        class Recognizer:
            def __init__(self): self.calls = 0
            def recognize_batch(self, images):
                self.calls += 1
                return [RecognitionResult("OK", 0.9) for _ in images]

        class ModelsStub:
            def __init__(self): self.localizer = Localizer(); self.detector = Detector(); self.recognizer = Recognizer()
            def document_localizer(self): return self.localizer
            def mrz_localizer(self): raise AssertionError("driving licence must not use MRZ localization")
            def text_detector(self): return self.detector
            def text_recognizer(self): return self.recognizer

        settings = Settings.from_env()
        image = Path("annotation_input/driving_licenses/test_license_canonical.jpg")
        document = Document("driving_license", "test_license_canonical", (("image", image),), Path("annotations/evaluation_ground_truth.json"))
        models = ModelsStub()
        run_partial(settings, models, [document], "driving_license", "driving_license_localization_preparation_only")
        self.assertEqual(models.detector.calls, 0)
        run_partial(settings, models, [document], "driving_license", "driving_license_visible_ocr_only")
        self.assertGreater(models.detector.calls, 0)
        self.assertGreater(models.recognizer.calls, 0)


if __name__ == "__main__":
    unittest.main()
