import unittest

from benchmarks.maintained.verification_benchmark import _accuracy
from benchmarks.maintained.verification_batch_size_benchmark import (
    _baseline,
    _individual_candidates,
    _output_differences,
)


class VerificationBenchmarkTests(unittest.TestCase):
    def test_accuracy_separates_statuses_and_failed_requests(self):
        rows = [
            {"request_status": "ok", "status": "match", "raw_exact_match": True, "score": 1.0},
            {"request_status": "ok", "status": "likely_match", "raw_exact_match": False, "score": 0.9},
            {"request_status": "ok", "status": "mismatch", "raw_exact_match": False, "score": 0.4},
            {"request_status": "ok", "status": "not_found", "raw_exact_match": False, "score": 0.0},
            {"request_status": "request_failed", "status": "request_failed", "raw_exact_match": False, "score": None},
        ]
        result = _accuracy(rows)
        self.assertEqual(5, result["total_truth_fields"])
        self.assertEqual(4, result["evaluated_fields"])
        self.assertEqual(1, result["normalized_exact_matches"])
        self.assertEqual(1, result["accepted_fuzzy_matches"])
        self.assertEqual(1, result["mismatches"])
        self.assertEqual(1, result["not_found"])
        self.assertEqual(0.8, result["field_verification_coverage"])

    def test_internal_sweep_includes_current_and_partition_boundaries(self):
        candidates = _individual_candidates(_baseline())
        names = {candidate.name for candidate in candidates}
        self.assertIn("baseline", names)
        self.assertTrue({"text_recognition-4", "text_recognition-8", "text_recognition-16", "text_recognition-32", "text_detection-2"} <= names)
        self.assertNotIn("text_recognition-64", names)
        self.assertNotIn("VERIFICATION_LOCALIZATION_BATCH_SIZE", {key for candidate in candidates for key in candidate.overrides})
        self.assertNotIn("VERIFICATION_MRZ_RECOGNITION_BATCH_SIZE", {key for candidate in candidates for key in candidate.overrides})

    def test_ocr_difference_report_includes_text_confidence_and_geometry(self):
        baseline = {"p_1": {"lines": [{"text": "A", "confidence": 0.8, "bbox": [1, 2, 3, 4], "line_id": "0"}]}}
        candidate = {"p_1": {"lines": [{"text": "B", "confidence": 0.7, "bbox": [1, 2, 4, 4], "line_id": "0"}]}}
        differences = _output_differences(baseline, candidate, "text_recognition-4")
        self.assertEqual({"text", "confidence", "bbox"}, {row["property"] for row in differences})


if __name__ == "__main__":
    unittest.main()
