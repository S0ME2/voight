from dataclasses import replace
from io import BytesIO
import unittest
import zipfile
from unittest.mock import patch

import cv2
import numpy as np

from app.api.v1 import _driving_field_results, _id_archive, _image_inputs, _run_batch, _safe_entries
from app.artifacts import ArtifactSettings
from app.config import Settings
from app.contracts import DocumentType, ErrorCode
from app.inference.batch import ProfileBatchOutcome
from app.main import create_app
from app.uploads import document_from_bytes
from app.workflows import WorkflowResult


def image_bytes() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def archive(entries: dict[str, bytes]) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w") as zipped:
        for name, data in entries.items():
            zipped.writestr(name, data)
    return output.getvalue()


class Runner:
    def __init__(self):
        self.calls = 0
        self.job_count = 0

    def run(self, jobs):
        self.calls += 1
        self.job_count = len(jobs)
        outcomes = []
        for job in jobs:
            names = job.profile.field_rois
            report = {
                "field_raw_text": {name: [] for name in names},
                "field_confidences": {name: None for name in names},
                "field_bounding_boxes": {name: None for name in names},
                "validation_warnings": [],
                "timings": {"total_seconds": 0.01},
            }
            outcomes.append(ProfileBatchOutcome(job.item_id, result=({name: None for name in names}, report)))
        return outcomes, {}


class Models:
    def __init__(self):
        self.runner = Runner()

    def document_aligner(self):
        return lambda **_: np.array([[0, 0], [15, 0], [15, 15], [0, 15]])

    def mrz_scanner(self):
        return lambda image, **_: {"mrz_polygon": [[0, 8], [15, 8], [15, 12], [0, 12]]}

    def profile_batch_runner(self):
        return self.runner


class V1TransportTests(unittest.TestCase):
    def setUp(self):
        base = Settings.from_env()
        self.settings = replace(base, artifacts=ArtifactSettings(False, base.artifacts.directory))

    def test_id_zip_pairs_nested_directories_by_filename(self):
        payload = archive({"a/card-2/front.jpg": image_bytes(), "a/card-2/back.jpg": image_bytes()})
        inputs = _id_archive(document_from_bytes(payload, "cards.zip"), self.settings)
        self.assertEqual(1, len(inputs))
        self.assertEqual(("front.jpg", "back.jpg"), tuple(file.filename for file in inputs[0].files))

    def test_id_zip_rejects_missing_or_duplicate_sides(self):
        missing = archive({"card/front.jpg": image_bytes()})
        with self.assertRaisesRegex(Exception, "requires front and back"):
            _id_archive(document_from_bytes(missing, "cards.zip"), self.settings)
        duplicate = archive({"card/front.jpg": image_bytes(), "card/front.png": image_bytes(), "card/back.jpg": image_bytes()})
        with self.assertRaisesRegex(Exception, "duplicate front"):
            _id_archive(document_from_bytes(duplicate, "cards.zip"), self.settings)

    def test_unsafe_zip_path_is_rejected(self):
        payload = archive({"../front.jpg": image_bytes()})
        with self.assertRaisesRegex(Exception, "unsafe path"):
            _safe_entries(document_from_bytes(payload, "bad.zip"), self.settings)

    def test_zip_limits_and_encryption_are_rejected(self):
        limited = replace(self.settings, batch=replace(self.settings.batch, max_archive_uncompressed_bytes=20))
        payload = archive({"card/front.jpg": b"x" * 21})
        with self.assertRaisesRegex(Exception, "exceeds BATCH_MAX_ARCHIVE"):
            _safe_entries(document_from_bytes(payload, "large.zip"), limited)

        class EncryptedEntry:
            filename = "card/front.jpg"
            file_size = 1
            flag_bits = 1

            def is_dir(self):
                return False

        class EncryptedZip:
            def __enter__(self):
                return self

            def __exit__(self, *_):
                return False

            def infolist(self):
                return [EncryptedEntry()]

        with patch("app.api.v1.zipfile.ZipFile", return_value=EncryptedZip()):
            with self.assertRaisesRegex(Exception, "Encrypted ZIP"):
                _safe_entries(document_from_bytes(payload, "encrypted.zip"), self.settings)

    def test_passport_batch_calls_shared_runner_once_and_keeps_invalid_sibling(self):
        inputs = _image_inputs(
            DocumentType.PASSPORT,
            [document_from_bytes(image_bytes(), "good.jpg"), document_from_bytes(b"not-image", "bad.jpg")],
        )
        models = Models()
        response = _run_batch(inputs, models, self.settings)
        self.assertEqual(1, models.runner.calls)
        self.assertEqual(1, models.runner.job_count)
        self.assertTrue(response.items[0].success)
        self.assertEqual(ErrorCode.INVALID_UPLOAD, response.items[1].error.code)

    def test_all_frozen_routes_are_declared_with_contract_schemas(self):
        schema = create_app(self.settings).openapi()
        paths = schema["paths"]
        for path in (
            "/v1/ocr/passport", "/v1/ocr/id-card", "/v1/ocr/driving-license",
            "/v1/ocr/passport/batch", "/v1/ocr/id-card/batch", "/v1/ocr/driving-license/batch",
            "/v1/health/live", "/v1/health/ready",
        ):
            self.assertIn(path, paths)
        self.assertEqual("#/components/schemas/OcrBatchResponse", paths["/v1/ocr/passport/batch"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"])
        request = paths["/v1/ocr/passport/batch"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        body = schema["components"]["schemas"][request["$ref"].split("/")[-1]]
        self.assertEqual("binary", body["properties"]["images"]["items"]["format"])

    def test_driving_derived_fields_reuse_their_source_evidence(self):
        fields = _driving_field_results(
            {"birth_place": "Toshloq", "birth_date": "19.10.2005"},
            {"field_raw_text": {"birth_place_and_date": ["3. TOSHLOQ 19.10.2005"]}, "field_confidences": {"birth_place_and_date": None}, "field_bounding_boxes": {"birth_place_and_date": None}},
        )
        self.assertEqual(["3. TOSHLOQ 19.10.2005"], fields["birth_place"].raw_text)
        self.assertEqual([], fields["birth_date"].raw_text)

    def test_identity_response_includes_mrz(self):
        models = Models()
        inputs = _image_inputs(DocumentType.PASSPORT, [document_from_bytes(image_bytes(), "passport.jpg")])
        mrz = "P<UZBCITIZEN<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<\n0000000000UZB0000000M00000000000000000000000"
        with patch("app.api.v1.run_document_mrz", return_value=WorkflowResult(mrz, {})):
            response = _run_batch(inputs, models, self.settings)
        self.assertEqual(["P<UZBCITIZEN<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<", "0000000000UZB0000000M00000000000000000000000"], response.items[0].result.mrz.raw_lines)


if __name__ == "__main__":
    unittest.main()
