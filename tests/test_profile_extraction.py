import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from app.artifacts import ArtifactWriter
from app.documents.driving_license_fields import parse_fields
from app.pipeline import RegionProfile, extract_profile, load_region_profile
from app.roi import assign_tokens_to_rois


class ProfileExtractionTests(unittest.TestCase):
    def setUp(self):
        self.writer = ArtifactWriter(Path("/unused"), "", "", False)

    def test_profile_is_validated_and_loaded_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            crop = root / "crop.json"
            rois = root / "rois.json"
            crop.write_text(
                json.dumps({"data_crop": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}})
            )
            rois.write_text(
                json.dumps({"field": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}})
            )
            load_region_profile.cache_clear()
            first = load_region_profile(crop, rois)
            crop.write_text("not json")
            self.assertIs(first, load_region_profile(crop, rois))

    def test_stub_pipeline_geometry_assignment_confidence_and_warnings(self):
        image = np.zeros((50, 100, 3), dtype=np.uint8)
        profile = RegionProfile(
            {"x1": 0.2, "y1": 0.2, "x2": 0.8, "y2": 0.8},
            {
                "present": {"x1": 0, "y1": 0, "x2": 0.5, "y2": 1},
                "missing": {"x1": 0.5, "y1": 0, "x2": 1, "y2": 1},
            },
        )
        observed = {}

        def detect(padded):
            observed["detector_shape"] = padded.shape
            return [[5, 5], [104, 5], [104, 54], [5, 54]]

        def recognize(crop):
            observed["crop_shape"] = crop.shape
            return [
                {"text": "A", "score": 0.8, "x1": 6, "y1": 3, "x2": 18, "y2": 9},
                {"text": "B", "score": 0.6, "x1": 12, "y1": 6, "x2": 24, "y2": 12},
            ]

        def parse(assignments):
            values = {
                field: " ".join(token["text"] for token in tokens) or None
                for field, tokens in assignments.items()
            }
            return values, {field: value or "" for field, value in values.items()}

        extracted, report = extract_profile(
            image,
            profile,
            detect,
            recognize,
            parse,
            lambda fields: ["Missing required field: missing"]
            if not fields["missing"]
            else [],
            self.writer,
            canonical_width=100,
            canonical_height=50,
            padding=5,
            min_overlap=0.5,
        )

        self.assertEqual((60, 110, 3), observed["detector_shape"])
        self.assertEqual((30, 60, 3), observed["crop_shape"])
        self.assertEqual({"present": "A B", "missing": None}, extracted)
        self.assertEqual(2, report["ocr_token_count"])
        self.assertEqual(0, report["unassigned_token_count"])
        self.assertAlmostEqual(0.7, report["field_confidences"]["present"]["score"])
        self.assertEqual("ocr_token_mean", report["field_confidences"]["present"]["source"])
        self.assertFalse(
            report["field_confidences"]["present"]["calibrated_probability"]
        )
        self.assertEqual(
            {"x1": 0.26, "y1": 0.26, "x2": 0.44, "y2": 0.44},
            report["field_bounding_boxes"]["present"],
        )
        self.assertIsNone(report["field_confidences"]["missing"])
        self.assertEqual(
            ["Missing required field: missing"], report["validation_warnings"]
        )

    def test_driving_license_parser_behavior_uses_shared_engine(self):
        profile = load_region_profile(
            Path("config/driving_license/data_crop.json"),
            Path("config/driving_license/field_rois_crop.json"),
        )
        image = np.zeros((630, 1000, 3), dtype=np.uint8)
        aligner = lambda _image: [
            [100, 100],
            [1099, 100],
            [1099, 729],
            [100, 729],
        ]
        tokens = lambda _crop: [
            {"text": "1. KARIMOV", "score": 0.9, "x1": 10, "y1": 10, "x2": 200, "y2": 40, "center_x": 105, "center_y": 25, "height": 30},
            {"text": "2. ALI", "score": 0.8, "x1": 10, "y1": 35, "x2": 200, "y2": 65, "center_x": 105, "center_y": 50, "height": 30},
        ]
        extracted, report = extract_profile(
            image,
            profile,
            aligner,
            tokens,
            parse_fields,
            lambda _fields: [],
            self.writer,
            canonical_width=1000,
            canonical_height=630,
            padding=100,
            min_overlap=0.3,
        )
        self.assertEqual("1. KARIMOV", extracted["surname"])
        self.assertEqual("2. ALI", extracted["given_names"])
        self.assertIsNone(extracted["patronymic"])
        self.assertIsNone(extracted["birth_place"])
        self.assertIsNone(extracted["birth_date"])
        self.assertEqual(0.9, report["field_confidences"]["surname"]["score"])
        self.assertEqual([], report["validation_warnings"])

    def test_token_center_must_be_inside_its_field(self):
        assignments, unassigned = assign_tokens_to_rois(
            [{"text": "label", "x1": 0, "y1": 0, "x2": 10, "y2": 10, "center_x": 4, "center_y": 5}],
            {"field": {"x1": 0.5, "y1": 0, "x2": 1, "y2": 1}},
            10,
            10,
            0.0,
        )
        self.assertEqual([], assignments["field"])
        self.assertEqual(["label"], [token["text"] for token in unassigned])

    def test_driving_license_annotation_labels_feed_canonical_fields(self):
        def token(text, y):
            return [{"text": text, "center_x": 10, "center_y": y, "x1": 0}]

        extracted, raw = parse_fields({
            "surname": token("1. QOBULOV", 1),
            "name": token("2. HUSNIDDIN", 2),
            "patronymic": token("3. O'G'LI", 3),
            "place_of_birth": token("3. TOSHLOQ", 3),
            "date_of_birth": token("19.10.2005", 3),
            "date_of_issue": token("4a. 17.02.2026", 4),
            "date_of_expiry": token("4b. 17.02.2036", 5),
            "place_of_issue": token("4c. TERMIZ DXM", 6),
            "id_number": token("4d. 51910056970036", 7),
            "id_number_2": token("5. AG2742395", 8),
            "place_of_living": token("8. FARGONA", 9),
            "types": token("9. B", 10),
            "serial_number": token("DL0008047407", 11),
        })
        self.assertEqual("2. HUSNIDDIN", extracted["given_names"])
        self.assertEqual("3. O'G'LI", extracted["patronymic"])
        self.assertEqual("3. TOSHLOQ", extracted["birth_place"])
        self.assertEqual("19.10.2005", extracted["birth_date"])
        self.assertEqual("4a. 17.02.2026", extracted["issue_date"])
        self.assertEqual("4b. 17.02.2036", extracted["expiry_date"])
        self.assertEqual("4d. 51910056970036", extracted["personal_id"])
        self.assertEqual("5. AG2742395", extracted["license_number"])
        self.assertEqual("9. B", extracted["categories"])
        self.assertEqual("2. HUSNIDDIN", raw["given_names"])


if __name__ == "__main__":
    unittest.main()
