import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.artifacts import ArtifactSettings
from app.config import Settings
from app.inference.batch import OcrBatchResult
from app.main import create_app
from app.models import Models
from app.verification import VerificationLine, VerificationThresholds, verify_fields


def image_bytes():
    ok, encoded = cv2.imencode(".jpg", np.zeros((8, 12, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


class FakeOcr:
    def __init__(self):
        self.sample_batches = []
        self.diagnostics = {}

    def run(self, samples):
        self.sample_batches.append(list(samples))
        tokens = {
            sample.item_id: [{"text": f"{sample.item_id}-line", "score": 0.98}]
            for sample in samples
        }
        return OcrBatchResult(tokens=tokens, errors={}, diagnostics=self.diagnostics)


class FakeModels:
    def __init__(self, ocr):
        self.ocr = ocr

    def verification_ocr(self):
        return self.ocr

    def preload(self):
        pass

    def close(self):
        pass


class VerificationMatcherTests(unittest.TestCase):
    def test_merged_label_and_value_yield_token_candidate(self):
        result = verify_fields([VerificationLine("ISMI / GIVEN NAMES JOHN", 0.9)], {"given_names": "JOHN"})
        field = result["fields"]["given_names"]
        self.assertEqual("match", field["status"])
        self.assertEqual("JOHN", field["detected"])
        self.assertEqual(["visible_ocr"], [item["source"] for item in field["evidence"]])
        self.assertEqual(["0"], [item["line_id"] for item in field["evidence"]])

    def test_merged_place_and_date_yield_two_semantic_spans(self):
        result = verify_fields(
            [VerificationLine("TOSHLOQ TUMANI 19.10.2005", 0.9)],
            {"birth_place": "3. TOSHLOQ TUMANI", "birth_date": "19.10.2005"},
        )
        self.assertEqual("match", result["fields"]["birth_place"]["status"])
        self.assertEqual("match", result["fields"]["birth_date"]["status"])
        self.assertNotEqual(result["fields"]["birth_place"]["evidence"][0]["span"], result["fields"]["birth_date"]["evidence"][0]["span"])

    def test_geometry_assembles_multiline_address_out_of_reading_order(self):
        result = verify_fields(
            [
                VerificationLine("TUMANI, ISTIQLOL MFY", 0.9, line_id="b", bbox=(10, 130, 220, 145), reading_order=3),
                VerificationLine("FARG'ONA VILOYATI, TOSHLOQ", 0.9, line_id="a", bbox=(10, 100, 260, 115), reading_order=1),
                VerificationLine("UNRELATED", 0.9, line_id="x", bbox=(500, 100, 650, 115), reading_order=2),
            ],
            {"address": "8. FARG'ONA VILOYATI, TOSHLOQ TUMANI, ISTIQLOL MFY"},
        )
        field = result["fields"]["address"]
        self.assertEqual("match", field["status"])
        self.assertEqual(["a", "b"], [item["line_id"] for item in field["evidence"]])

    def test_field_semantics_beat_digits_in_expected_text(self):
        from app.verification import _kind

        self.assertEqual("text", _kind("address", "8. TOSHKENT 16"))
        self.assertEqual("text", _kind("birth_place", "3. TOSHLOQ"))
        self.assertEqual("identifier", _kind("personal_id", "123456"))
        result = verify_fields([VerificationLine("8. TOSHKENT 16", 0.9)], {"address": "8. TOSHKENT 16"})
        self.assertEqual("match", result["fields"]["address"]["status"])

    def test_strict_identifiers_and_dates_do_not_accept_close_values(self):
        result = verify_fields(
            [VerificationLine("060300031283", 0.9), VerificationLine("19.10.2006", 0.9)],
            {"serial_number": "060300031233", "birth_date": "19.10.2005"},
        )
        self.assertEqual("mismatch", result["fields"]["serial_number"]["status"])
        self.assertEqual("mismatch", result["fields"]["birth_date"]["status"])

    def test_text_can_contain_legitimate_embedded_numbers(self):
        result = verify_fields([VerificationLine("TOSHKENT KO'CHASI 16V, 131", 0.9)], {"address": "TOSHKENT KO'CHASI 16V, 131"})
        self.assertEqual("match", result["fields"]["address"]["status"])

    def test_shared_ocr_line_uses_non_overlapping_token_spans(self):
        result = verify_fields([VerificationLine("JOHN 060300031233", 0.9)], {"name": "JOHN", "serial_number": "060300031233"})
        self.assertEqual("match", result["fields"]["name"]["status"])
        self.assertEqual("match", result["fields"]["serial_number"]["status"])

    def test_overlapping_spans_cannot_be_reused(self):
        result = verify_fields([VerificationLine("JOHN", 0.9)], {"name": "JOHN", "surname": "JOHN"})
        self.assertEqual(1, sum(field["status"] == "match" for field in result["fields"].values()))

    def test_unrelated_or_distant_blocks_are_not_assembled(self):
        result = verify_fields(
            [
                VerificationLine("FARG'ONA VILOYATI", 0.9, bbox=(10, 100, 180, 115)),
                VerificationLine("UNRELATED", 0.9, bbox=(300, 118, 420, 133)),
                VerificationLine("ISTIQLOL MFY", 0.9, bbox=(10, 300, 120, 315)),
            ],
            {"address": "FARG'ONA VILOYATI ISTIQLOL MFY"},
        )
        self.assertNotEqual("match", result["fields"]["address"]["status"])

    def test_valid_mrz_is_grouped_and_used_as_evidence(self):
        result = verify_fields(
            [
                VerificationLine("P<UZBCITIZEN<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<", 0.9, line_id="mrz-1"),
                VerificationLine("0000000000UZB0000000M00000000000000000000000", 0.9, line_id="mrz-2"),
            ],
            {"surname": "CITIZEN", "given_names": "JOHN", "passport_number": "000000000"},
            document_type="passport",
        )
        self.assertEqual("mrz", result["fields"]["surname"]["source"])
        self.assertEqual("mrz", result["fields"]["passport_number"]["source"])
        self.assertEqual({"mrz-1", "mrz-2"}, {item["line_id"] for field in result["fields"].values() for item in field["evidence"]})

    def test_valid_id_card_mrz_supplies_identifier_name_and_dates(self):
        from tests.test_v1_api import ID_MRZ

        result = verify_fields(
            [VerificationLine(value, 0.9, line_id=f"id-mrz-{index}") for index, value in enumerate(ID_MRZ.splitlines())],
            {
                "surname": "EGAMOVA",
                "name": "IRODA",
                "card_number": "AD6632763",
                "pinfl": "41103741390036",
                "date_of_birth": "11.03.1974",
                "date_of_expiry": "27.03.2034",
            },
            document_type="id_card",
        )
        self.assertEqual({"match": 6, "likely_match": 0, "mismatch": 0, "not_found": 0}, result["summary"])
        self.assertTrue(all(field["source"] == "mrz" for field in result["fields"].values()))

    def test_visible_exact_value_wins_over_conflicting_mrz_evidence(self):
        result = verify_fields(
            [
                VerificationLine("VISIBLE", 0.9, line_id="visible"),
                VerificationLine("P<UZBCITIZEN<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<", 0.9, line_id="mrz-1"),
                VerificationLine("0000000000UZB0000000M00000000000000000000000", 0.9, line_id="mrz-2"),
            ],
            {"surname": "VISIBLE"},
            document_type="passport",
        )
        self.assertEqual("visible_ocr", result["fields"]["surname"]["source"])

    def test_missing_geometry_still_reports_line_provenance(self):
        result = verify_fields([VerificationLine("JOHN", 0.9)], {"name": "JOHN"})
        self.assertEqual("0", result["fields"]["name"]["evidence"][0]["line_id"])
        self.assertIsNone(result["fields"]["name"]["evidence"][0]["bbox"])

    def test_id_card_sides_are_not_cross_assembled(self):
        result = verify_fields(
            [
                VerificationLine("FRONT", 0.9, "front", side="front", bbox=(0, 0, 60, 10)),
                VerificationLine("BACK", 0.9, "back", side="back", bbox=(0, 12, 60, 22)),
            ],
            {"surname": "FRONT", "name": "BACK", "address": "FRONT BACK"},
            document_type="id_card",
        )
        self.assertEqual("front", result["fields"]["surname"]["source"])
        self.assertEqual("back", result["fields"]["name"]["source"])
        self.assertNotEqual("match", result["fields"]["address"]["status"])

    def test_normalization_dates_split_lines_and_name_typo(self):
        result = verify_fields(
            [
                VerificationLine("  abdullaev  ", 0.9),
                VerificationLine("AKMAI", 0.9),
                VerificationLine("01/02/1995", 0.9),
                VerificationLine("AA 1234567", 0.9),
            ],
            {
                "surname": "ABDULLAEV",
                "name": "AKMAL",
                "date_of_birth": "01.02.1995",
                "passport_number": "AA1234567",
            },
        )
        self.assertEqual("match", result["fields"]["surname"]["status"])
        self.assertEqual("likely_match", result["fields"]["name"]["status"])
        self.assertEqual("match", result["fields"]["date_of_birth"]["status"])
        self.assertEqual("match", result["fields"]["passport_number"]["status"])

        split = verify_fields([VerificationLine("AK", 0.9), VerificationLine("MAL", 0.9)], {"name": "AKMAL"})
        self.assertEqual("match", split["fields"]["name"]["status"])

    def test_identifiers_dates_short_values_and_missing_are_conservative(self):
        result = verify_fields(
            [VerificationLine("AA1234568", 0.9), VerificationLine("02.02.1995", 0.9), VerificationLine("AB", 0.9)],
            {"passport_number": "AA1234567", "date_of_birth": "01.02.1995", "code": "AC"},
        )
        self.assertEqual("mismatch", result["fields"]["passport_number"]["status"])
        self.assertEqual("mismatch", result["fields"]["date_of_birth"]["status"])
        self.assertEqual("mismatch", result["fields"]["code"]["status"])

        missing = verify_fields([], {"surname": "ABDULLAEV", "passport_number": "AA1234567"})
        self.assertEqual({"not_found": 2, "match": 0, "likely_match": 0, "mismatch": 0}, missing["summary"])

    def test_global_assignment_does_not_reuse_one_line_and_ignores_unrelated_text(self):
        result = verify_fields(
            [VerificationLine("JOHN", 0.9), VerificationLine("AA1234568", 0.9), VerificationLine("UNRELATED", 0.9)],
            {"name": "JOHN", "given_name": "JOHN", "passport_number": "AA1234567"},
        )
        statuses = [field["status"] for field in result["fields"].values()]
        self.assertEqual(1, statuses.count("match"))
        self.assertEqual(1, statuses.count("not_found"))
        self.assertEqual(1, statuses.count("mismatch"))

    def test_duplicate_expected_values_cannot_reuse_one_ocr_line(self):
        result = verify_fields(
            [VerificationLine("AKMAL", 0.9)],
            {"surname": "AKMAL", "name": "AKMAL"},
        )
        self.assertEqual(1, sum(value["status"] == "match" for value in result["fields"].values()))
        self.assertEqual(1, sum(value["detected"] is not None for value in result["fields"].values()))

    def test_cross_field_value_competes_for_one_candidate(self):
        result = verify_fields(
            [VerificationLine("ABDULLAEV", 0.9), VerificationLine("AKMAL", 0.9)],
            {"surname": "AKMAL", "name": "AKMAL"},
        )
        self.assertEqual(1, sum(value["status"] == "match" for value in result["fields"].values()))
        self.assertEqual(1, sum(value["detected"] == "AKMAL" for value in result["fields"].values()))

    def test_id_card_source_is_retained(self):
        result = verify_fields([VerificationLine("ABDULLAEV", 0.9, "front")], {"surname": "ABDULLAEV"})
        self.assertEqual("front", result["fields"]["surname"]["source"])

    def test_irrelevant_punctuation_and_identifier_separators_are_normalized(self):
        result = verify_fields(
            [VerificationLine("O'NEIL", 0.9), VerificationLine("AA-1234567", 0.9)],
            {"surname": "ONEIL", "passport_number": "AA1234567"},
        )
        self.assertEqual("match", result["fields"]["surname"]["status"])
        self.assertEqual("match", result["fields"]["passport_number"]["status"])

    def test_thresholds_are_centralized_and_overridable(self):
        result = verify_fields(
            [VerificationLine("ABDULAEV", 0.9)],
            {"surname": "ABDULLAEV"},
            VerificationThresholds(likely_name_score=0.99),
        )
        self.assertEqual("mismatch", result["fields"]["surname"]["status"])


class VerificationTransportTests(unittest.TestCase):
    def setUp(self):
        base = Settings.from_env()
        self.settings = replace(base, artifacts=ArtifactSettings(False, base.artifacts.directory))
        self.ocr = FakeOcr()
        self.models = FakeModels(self.ocr)
        with patch("app.main.Models", return_value=self.models):
            self.application = create_app(self.settings)

    def test_full_document_routes_use_shared_batch_ocr_and_preserve_id_sides(self):
        payload = image_bytes()
        with TestClient(self.application) as client:
            passport = client.post("/verification/passport/ocr", files={"image": ("p.jpg", payload, "image/jpeg")})
            card = client.post(
                "/verification/id-card/ocr",
                files={"front": ("front.jpg", payload, "image/jpeg"), "back": ("back.jpg", payload, "image/jpeg")},
            )
            batch = client.post(
                "/verification/driving-licence/ocr/batch",
                files=[("images", ("one.jpg", payload, "image/jpeg")), ("images", ("two.jpg", payload, "image/jpeg"))],
            )
        self.assertEqual(200, passport.status_code)
        self.assertEqual(200, card.status_code)
        self.assertEqual(["front", "back"], list(card.json()))
        self.assertEqual(200, batch.status_code)
        self.assertEqual(2, batch.json()["succeeded"])
        self.assertEqual([1, 2, 2], [len(samples) for samples in self.ocr.sample_batches])
        self.assertTrue(all(sample.recognition_rois is None for samples in self.ocr.sample_batches for sample in samples))

    def test_check_routes_and_invalid_payloads(self):
        with TestClient(self.application) as client:
            response = client.post(
                "/verification/passport/check",
                json={"ocr": {"lines": [{"text": "abdullaev", "confidence": 0.98}]}, "fields": {"surname": "ABDULLAEV"}},
            )
            card = client.post(
                "/verification/id-card/check",
                json={
                    "ocr": {"front": [{"text": "ABDULLAEV", "confidence": 0.98}], "back": []},
                    "fields": {"surname": "ABDULLAEV", "pinfl": "41103741390036"},
                },
            )
            invalid = client.post("/verification/passport/check", json={"ocr": {"lines": []}, "fields": {}})
            bad_image = client.post("/verification/passport/ocr", files={"image": ("p.jpg", b"not image", "image/jpeg")})
        self.assertEqual("match", response.json()["fields"]["surname"]["status"])
        self.assertEqual("front", card.json()["fields"]["surname"]["source"])
        self.assertEqual("not_found", card.json()["fields"]["pinfl"]["status"])
        self.assertEqual(422, invalid.status_code)
        self.assertEqual(422, bad_image.status_code)

    def test_verification_check_writes_debug_artifacts(self):
        with TemporaryDirectory() as directory:
            settings = replace(
                Settings.from_env(),
                artifacts=ArtifactSettings(True, Path(directory)),
            )
            with patch("app.main.Models", return_value=self.models):
                application = create_app(settings)
            with TestClient(application) as client:
                response = client.post(
                    "/verification/passport/check",
                    json={
                        "ocr": {"lines": [{"text": "ABDULLAEV", "confidence": 0.98}]},
                        "fields": {"surname": "ABDULLAEV"},
                    },
                )

            self.assertEqual(200, response.status_code)
            run = next((Path(directory) / "verification_check").iterdir())
            self.assertEqual(
                {"00_request.json", "01_normalized_ocr_lines.json", "02_matcher_result.json", "03_response.json", "04_timing.json"},
                {path.name for path in run.iterdir()},
            )

    def test_verification_ocr_writes_artifacts_when_logging_is_enabled(self):
        with TemporaryDirectory() as directory:
            settings = replace(
                Settings.from_env(),
                artifacts=ArtifactSettings(True, Path(directory)),
            )
            self.ocr.diagnostics = {"text_detection": {"tensor_batch_sizes": [1]}}
            with patch("app.main.Models", return_value=self.models):
                application = create_app(settings)
            with TestClient(application) as client:
                response = client.post(
                    "/verification/id-card/ocr",
                    files={
                        "front": ("front.jpg", image_bytes(), "image/jpeg"),
                        "back": ("back.jpg", image_bytes(), "image/jpeg"),
                    },
                )

            run_directories = list((Path(directory) / "verification_batch").iterdir())
            self.assertEqual(200, response.status_code)
            self.assertEqual(["front", "back"], list(response.json()))
            self.assertEqual(1, len(run_directories))
            run = run_directories[0]
            self.assertEqual(
                {"text_detection": {"tensor_batch_sizes": [1]}},
                json.loads((run / "01_ocr_diagnostics.json").read_text()),
            )
            for side in ("001_front", "002_back"):
                self.assertTrue((run / side / "00_input.jpg").is_file())
                self.assertTrue((run / side / "01_ocr_tokens.json").is_file())

    def test_verification_ocr_does_not_write_artifacts_when_logging_is_disabled(self):
        with TemporaryDirectory() as directory:
            settings = replace(
                Settings.from_env(),
                artifacts=ArtifactSettings(False, Path(directory)),
            )
            with patch("app.main.Models", return_value=self.models):
                application = create_app(settings)
            with TestClient(application) as client:
                response = client.post(
                    "/verification/passport/ocr",
                    files={"image": ("passport.jpg", image_bytes(), "image/jpeg")},
                )
            self.assertEqual(200, response.status_code)
            self.assertEqual([], list(Path(directory).iterdir()))


class VerificationModelLifecycleTests(unittest.TestCase):
    def test_verification_ocr_is_cached_and_shares_text_models(self):
        settings = replace(Settings.from_env(), artifacts=ArtifactSettings(False, Settings.from_env().artifacts.directory))
        models = Models(settings)
        detector = object()
        recognizer = object()
        models.text_detector = lambda: detector
        models.text_recognizer = lambda: recognizer
        first = models.verification_ocr()
        self.assertIs(first, models.verification_ocr())
        self.assertIs(detector, first.detector)
        self.assertIs(recognizer, first.recognizer)

    def test_verification_ocr_uses_verification_limits_without_changing_global_runner(self):
        settings = replace(
            Settings.from_env(),
            verification=replace(
                Settings.from_env().verification,
                text_detection_batch_size=7,
                text_recognition_batch_size=9,
            ),
        )
        models = Models(settings)
        detector = object()
        recognizer = object()
        models.text_detector = lambda: detector
        models.text_recognizer = lambda: recognizer
        models.document_localizer = lambda: object()
        models.mrz_localizer = lambda: object()
        verification = models.verification_ocr()
        self.assertEqual(7, verification.detection_batch_size)
        self.assertEqual(9, verification.recognition_batch_size)
        self.assertEqual(settings.runtime.mrz_recognition_batch_size, verification.mrz_recognition_batch_size)
        self.assertEqual(settings.runtime.text_detection_batch_size, models.profile_batch_runner().ocr.detection_batch_size)
        self.assertEqual(settings.runtime.text_recognition_batch_size, models.profile_batch_runner().ocr.recognition_batch_size)


if __name__ == "__main__":
    unittest.main()
