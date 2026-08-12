import unittest

from scripts.validation.local import batch_evidence, extraction_evidence


class LocalValidationTests(unittest.TestCase):
    def test_one_example_and_synthetic_checks_are_explicit(self):
        evidence = extraction_evidence()
        self.assertEqual(13, evidence["baseline"]["passport"]["exact_matches"])
        self.assertEqual(12, evidence["baseline"]["id_card"]["exact_matches"])
        self.assertEqual({"rotation", "perspective", "background", "blur", "glare"}, set(evidence["synthetic_robustness"]))

    def test_batch_proof_contains_a_real_multi_sample_call(self):
        evidence = batch_evidence([1, 3], 1)
        self.assertTrue(evidence["true_batch_observed"])
        self.assertEqual([3], evidence["rows"][1]["true_batch"]["model_call_proof"]["text_detection"])


if __name__ == "__main__":
    unittest.main()
