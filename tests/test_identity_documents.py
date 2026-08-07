import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.artifacts import ArtifactWriter
from app.contracts import ErrorCode, ValidationStatus
from app.documents.identity import (
    DocumentPipelineError,
    extract_id_card,
    extract_passport,
)
from app.documents.mrz import check_digit, parse


ROOT = Path(__file__).resolve().parents[1]
TRUTH = json.loads((ROOT / "annotations/evaluation_ground_truth.json").read_text())["samples"]
PASSPORT_PROFILE = ROOT / "config/documents/uz_passport/profile.json"
ID_PROFILE = ROOT / "config/documents/uz_id_card/profile.json"
WRITER = ArtifactWriter(Path("/unused"), "", "", False)
PASSPORT_EXPECTED = {
    "type": "P", "country_code": "UZB", "passport_number": "000000000",
    "surname": "CITIZEN", "name": "JOHN", "patronymic": "DOE",
    "nationality": "UZBEKISTAN", "date_of_birth": "01.01.2000", "sex": "M",
    "place_of_birth": "TASHKENT", "date_of_issue": "01.01.2020",
    "date_of_expiry": "01.01.2030", "authority": "IIV",
}
FRONT_EXPECTED = {
    "surname": "EGAMOVA", "name": "IRODA", "patronymic": "IBROXIMOVNA",
    "date_of_birth": "11.03.1974", "date_of_issue": "28.03.2024",
    "date_of_expiry": "27.03.2034", "sex": "AYOL", "citizenship": "O'ZBEKISTON",
    "card_number": "AD6632763",
}
BACK_EXPECTED = {"pinfl": "41103741390036", "place_of_birth": "SHAXRIXON", "place_of_issue": "IIV3234"}
PASSPORT_MRZ = "P<UZBCITIZEN<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<\n0000000000UZB0000000M00000000000000000000000"
ID_MRZ = "I<UZBAD6632763841103741390036<\n7403116F3403277UZB<<<<<<<<<<<8\nEGAMOVA<<IRODA<<<<<<<<<<<<<<<<"


def full_document(image):
    height, width = image.shape[:2]
    return {
        "corners": [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
        "score": 0.91,
    }


def annotated_document(sample, score=0.91):
    size = sample["original_size"]
    corners = [
        [x * size["width"], y * size["height"]]
        for x, y in sample["corners"]
    ]
    return lambda _image: {"corners": corners, "score": score}


def tokens(profile_path, region, values, crop):
    profile = json.loads(profile_path.read_text())
    height, width = crop.shape[:2]
    output = []
    for index, (field, value) in enumerate(values.items()):
        roi = profile["regions"][region]["field_rois"][field]
        x1, y1 = roi["x1"] * width, roi["y1"] * height
        x2, y2 = roi["x2"] * width, roi["y2"] * height
        output.append(
            {
                "index": index,
                "text": value,
                "score": 0.8 + index / 100,
                "x1": x1 + 1,
                "y1": y1 + 1,
                "x2": x2 - 1,
                "y2": y2 - 1,
                "center_x": (x1 + x2) / 2,
                "center_y": (y1 + y2) / 2,
                "height": max(1, y2 - y1 - 2),
            }
        )
    return output


class IdentityDocumentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.passport = cv2.imread(str(ROOT / TRUTH["passport:passport.png"]["source"]))
        cls.front = cv2.imread(str(ROOT / TRUTH["id_card:uzbekistan_id_001:front"]["source"]))
        cls.back = cv2.imread(str(ROOT / TRUTH["id_card:uzbekistan_id_001:back"]["source"]))

    def test_passport_golden_visible_fields_and_mrz(self):
        expected = PASSPORT_EXPECTED
        result = extract_passport(
            self.passport,
            PASSPORT_PROFILE,
            annotated_document(TRUTH["passport:passport.png"]),
            lambda crop: tokens(PASSPORT_PROFILE, "data_page", expected, crop),
            lambda _image: PASSPORT_MRZ,
            WRITER,
        )

        self.assertEqual(expected, {name: field.value for name, field in result.fields.items()})
        self.assertEqual("data_page", result.fields["surname"].region)
        self.assertEqual(["CITIZEN"], result.fields["surname"].raw_text)
        self.assertEqual("ocr_token_mean", result.fields["surname"].confidence.source.value)
        self.assertEqual("000000000", result.mrz.fields["document_number"])
        self.assertEqual(ValidationStatus.PASSED, result.mrz.validations[0].status)
        self.assertEqual(0.91, result.document_confidence.score)

    def test_paired_id_golden_merges_sides_and_reports_truth_conflicts(self):
        front_expected = FRONT_EXPECTED
        back_expected = BACK_EXPECTED

        expected_by_region = iter((("front", front_expected), ("back", back_expected)))

        def recognize(crop):
            region, values = next(expected_by_region)
            return tokens(ID_PROFILE, region, values, crop)

        detections = iter(
            [
                annotated_document(TRUTH["id_card:uzbekistan_id_001:front"])(None),
                annotated_document(TRUTH["id_card:uzbekistan_id_001:back"])(None),
            ]
        )
        result = extract_id_card(
            self.front,
            self.back,
            ID_PROFILE,
            lambda _image: next(detections),
            recognize,
            lambda image: ID_MRZ if image is self.back else "",
            WRITER,
        )

        self.assertEqual({**front_expected, **back_expected}, {name: field.value for name, field in result.fields.items()})
        self.assertEqual("front", result.fields["card_number"].region)
        self.assertEqual("back", result.fields["pinfl"].region)
        validations = {item.code: item.status for item in result.validations}
        self.assertEqual(ValidationStatus.PASSED, validations["visible_mrz_card_number"])
        self.assertEqual(ValidationStatus.PASSED, validations["visible_mrz_pinfl"])
        self.assertEqual(ValidationStatus.PASSED, validations["visible_mrz_date_of_birth"])
        self.assertEqual(ValidationStatus.PASSED, validations["visible_mrz_date_of_expiry"])

    def test_missing_and_swapped_id_sides_are_structured(self):
        with self.assertRaises(DocumentPipelineError) as missing:
            extract_id_card(self.front, None, ID_PROFILE, full_document, lambda _crop: [], lambda _image: "", WRITER)
        self.assertEqual(ErrorCode.MISSING_DOCUMENT_SIDE, missing.exception.error.code)

        with self.assertRaises(DocumentPipelineError) as swapped:
            extract_id_card(
                self.back,
                self.front,
                ID_PROFILE,
                full_document,
                lambda _crop: [],
                lambda image: ID_MRZ if image is self.back else "",
                WRITER,
            )
        self.assertEqual(ErrorCode.INVALID_DOCUMENT, swapped.exception.error.code)
        self.assertNotIn("Traceback", swapped.exception.error.detail)

    def test_malformed_or_missing_mrz_and_partial_fields_do_not_crash(self):
        result = extract_passport(
            self.passport,
            PASSPORT_PROFILE,
            full_document,
            lambda crop: tokens(PASSPORT_PROFILE, "data_page", {"surname": "CITIZEN"}, crop),
            lambda _image: "P<UZB\nSHORT",
            WRITER,
        )
        self.assertEqual(ValidationStatus.FAILED, result.mrz.validations[0].status)
        self.assertIsNone(result.fields["passport_number"].value)
        self.assertIsNone(result.fields["passport_number"].confidence)
        self.assertIn("Missing required field: passport_number", result.warnings)

        missing = extract_passport(
            self.passport,
            PASSPORT_PROFILE,
            full_document,
            lambda _crop: [],
            lambda _image: (_ for _ in ()).throw(IndexError("no polygon")),
            WRITER,
        )
        self.assertEqual(ValidationStatus.NOT_RUN, missing.mrz.validations[0].status)
        self.assertIn("MRZ was not found", missing.warnings)

    def test_no_document_is_a_sanitized_item_error(self):
        with self.assertRaises(DocumentPipelineError) as failure:
            extract_passport(
                self.passport,
                PASSPORT_PROFILE,
                lambda _image: [],
                lambda _crop: [],
                lambda _image: "",
                WRITER,
            )
        self.assertEqual(ErrorCode.INVALID_DOCUMENT, failure.exception.error.code)
        self.assertEqual("data_page image could not be localized", failure.exception.error.detail)

    def test_icao_check_digits_and_visible_ocr_conflict(self):
        self.assertEqual("6", check_digit("740311"))
        parsed = parse(ID_MRZ, "id_card")
        self.assertEqual("EGAMOVA", parsed.fields["surname"])
        self.assertTrue(all(item.status == ValidationStatus.PASSED for item in parsed.validations))

        expected = {**PASSPORT_EXPECTED, "surname": "OTHER"}
        result = extract_passport(
            self.passport,
            PASSPORT_PROFILE,
            full_document,
            lambda crop: tokens(PASSPORT_PROFILE, "data_page", expected, crop),
            lambda _image: PASSPORT_MRZ,
            WRITER,
        )
        conflict = next(item for item in result.validations if item.code == "visible_mrz_surname")
        self.assertEqual(ValidationStatus.FAILED, conflict.status)


if __name__ == "__main__":
    unittest.main()
