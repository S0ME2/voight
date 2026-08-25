import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.api.v1 import _driving_field_results
from app.artifacts import ArtifactSettings
from app.config import Settings, TextModelSettings
from app.contracts import DocumentType
from app.inference.batch import ProfileBatchOutcome
from app.inference.contracts import LocalizationResult, RecognitionResult
from app.models import Models
from app.main import create_app


ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests/golden/pipeline.json"
PASSPORT_MRZ = "P<UZBCITIZEN<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<\n0000000000UZB0000000M00000000000000000000000"
ID_MRZ = "I<UZBAD6632763841103741390036<\n7403116F3403277UZB<<<<<<<<<<<8\nEGAMOVA<<IRODA<<<<<<<<<<<<<<<<"
PASSPORT_FIELDS = {
    "type": "P", "country_code": "UZB", "passport_number": "000000000",
    "surname": "CITIZEN", "name": "JOHN", "patronymic": "DOE",
    "nationality": "UZBEKISTAN", "date_of_birth": "01.01.2000", "sex": "M",
    "place_of_birth": "TASHKENT", "date_of_issue": "01.01.2020",
    "date_of_expiry": "01.01.2030", "authority": "IIV",
}
ID_FRONT_FIELDS = {
    "surname": "EGAMOVA", "name": "IRODA", "patronymic": "IBROXIMOVNA",
    "date_of_birth": "11.03.1974", "date_of_issue": "28.03.2024",
    "date_of_expiry": "27.03.2034", "sex": "AYOL", "citizenship": "O'ZBEKISTON",
    "card_number": "AD6632763",
}
ID_BACK_FIELDS = {"pinfl": "41103741390036", "place_of_birth": "SHAXRIXON", "place_of_issue": "IIV3234"}
DRIVING_FIELDS = {
    "surname": "DRIVER", "name": "JOHN", "patronymic": "DOE", "place_of_birth": "TASHKENT",
    "date_of_birth": "01.01.2000", "date_of_issue": "01.01.2020", "date_of_expiry": "01.01.2030",
    "place_of_issue": "IIV", "id_number": "AA1234567", "id_number_2": "BB7654321",
    "place_of_living": "TASHKENT", "types": "B", "serial_number": "UZ",
}


def _fixture(name: str) -> bytes:
    return (ROOT / name).read_bytes()


def _report(values: dict[str, str | None], fields: list[str]) -> dict:
    return {
        "field_raw_text": {field: [values[field]] if values.get(field) else [] for field in fields},
        "field_confidences": {field: {"score": 0.9, "source": "ocr_token_mean", "calibrated_probability": False} if values.get(field) else None for field in fields},
        "field_bounding_boxes": {field: {"x1": 0.1, "y1": 0.1, "x2": 0.2, "y2": 0.2} if values.get(field) else None for field in fields},
        "validation_warnings": [],
        "document_confidence": {"score": 0.95, "source": "document_detection", "calibrated_probability": False},
        "timings": {"total_seconds": 0.01, "model_seconds": 0.01},
    }


class StubLocalizer:
    def __init__(self): self.calls = []
    def localize_batch(self, images):
        self.calls.append(len(images))
        self.last_tensor_batch_size = len(images)
        return [LocalizationResult(np.asarray([[0, 0], [image.shape[1] - 1, 0], [image.shape[1] - 1, image.shape[0] - 1], [0, image.shape[0] - 1]], np.float32), 0.95) for image in images]


class StubDetector:
    def __init__(self): self.calls = []
    def detect_batch(self, images):
        self.calls.append(len(images))
        return []


class StubRecognizer:
    def __init__(self): self.calls = []; self.model = object()
    def recognize_batch(self, images):
        self.calls.append(len(images))
        return [RecognitionResult("stub", 0.9) for _ in images]


class GoldenRunner:
    def __init__(self, localizer, detector, recognizer):
        self.localizer, self.detector, self.recognizer = localizer, detector, recognizer
        self.calls = 0

    def run(self, jobs):
        self.calls += 1
        self.localizer.localize_batch([job.image for job in jobs])
        self.detector.detect_batch([job.image for job in jobs])
        self.recognizer.recognize_batch([job.image for job in jobs])
        outcomes = []
        for job in jobs:
            if job.item_id.endswith(":data_page"):
                values = PASSPORT_FIELDS
                fields = list(values)
                outcomes.append(ProfileBatchOutcome(job.item_id, (values, _report(values, fields)), PASSPORT_MRZ, True))
            elif job.item_id.endswith(":front"):
                outcomes.append(ProfileBatchOutcome(job.item_id, (ID_FRONT_FIELDS, _report(ID_FRONT_FIELDS, list(ID_FRONT_FIELDS))), None, False))
            elif job.item_id.endswith(":back"):
                outcomes.append(ProfileBatchOutcome(job.item_id, (ID_BACK_FIELDS, _report(ID_BACK_FIELDS, list(ID_BACK_FIELDS))), ID_MRZ, True))
            else:
                source = {
                    "surname": DRIVING_FIELDS["surname"], "given_names": DRIVING_FIELDS["name"], "patronymic": DRIVING_FIELDS["patronymic"],
                    "birth_place": DRIVING_FIELDS["place_of_birth"], "birth_date": DRIVING_FIELDS["date_of_birth"], "issue_date": DRIVING_FIELDS["date_of_issue"],
                    "expiry_date": DRIVING_FIELDS["date_of_expiry"], "issued_place": DRIVING_FIELDS["place_of_issue"], "personal_id": DRIVING_FIELDS["id_number"],
                    "license_number": DRIVING_FIELDS["id_number_2"], "address": DRIVING_FIELDS["place_of_living"], "categories": DRIVING_FIELDS["types"], "serial_number": DRIVING_FIELDS["serial_number"],
                }
                outcomes.append(ProfileBatchOutcome(job.item_id, (source, _report(source, list(source))), None, False))
        return outcomes, {"stub_models": {"localizer": self.localizer.calls, "detector": self.detector.calls, "recognizer": self.recognizer.calls}}


def _semantic(response: dict) -> dict:
    def result(value):
        if not value:
            return {"success": False, "error": value.get("error")}
        result = value["result"]
        return {
            "success": True,
            "document_type": result["document_type"],
            "layout": result["layout"],
            "fields": result["fields"],
            "mrz": result.get("mrz"),
            "validations": result["validations"],
            "warnings": result["warnings"],
        }
    if "result" in response:
        return result(response)
    return {
        "total": response["total"], "succeeded": response["succeeded"], "failed": response["failed"],
        "items": [result(item) if item["success"] else {"success": False, "error": item["error"]} for item in response["items"]],
    }


def build_golden() -> dict:
    with patch.dict(os.environ, {"MODEL_DIR": ""}, clear=False):
        settings = Settings.from_env()
    settings = replace_artifacts(settings)
    settings = replace(settings, models=replace(settings.models, text_recognizer=TextModelSettings("golden", "stub")))
    factories = {"golden": lambda _selection: StubRecognizer()}
    models = Models(settings, text_recognizer_factories=factories)
    recognizer = models.text_recognizer()
    localizer, detector = StubLocalizer(), StubDetector()
    runner = GoldenRunner(localizer, detector, recognizer)
    models.profile_batch_runner = lambda: runner
    with patch("app.main.Models", return_value=models):
        application = create_app(settings)
    passport = _fixture("annotation_input/passports/passport.png")
    front = _fixture("annotation_input/id_cards/uzbekistan_id_001/front.png")
    back = _fixture("annotation_input/id_cards/uzbekistan_id_001/back.png")
    driving = _fixture("annotation_input/driving_licenses/test_license_canonical.jpg")
    with TestClient(application) as client:
        responses = {
            "passport": client.post("/v1/ocr/passport", files={"image": ("passport.png", passport, "image/png")}),
            "id_card": client.post("/v1/ocr/id-card", files={"front": ("front.png", front, "image/png"), "back": ("back.png", back, "image/png")}),
            "driving_license": client.post("/v1/ocr/driving-license", files={"image": ("license.jpg", driving, "image/jpeg")}),
            "per_item_errors": client.post("/v1/ocr/passport/batch", files=[("images", ("good.png", passport, "image/png")), ("images", ("bad.png", b"bad", "image/png"))]),
        }
    assert all(response.status_code == 200 for response in responses.values())
    return {name: _semantic(response.json()) for name, response in responses.items()}


def replace_artifacts(settings):
    return replace(settings, artifacts=ArtifactSettings(False, settings.artifacts.directory))


class GoldenPipelineTests(unittest.TestCase):
    def test_fixture_routes_match_committed_semantic_golden(self):
        expected = json.loads(GOLDEN.read_text())
        actual = build_golden()
        self.assertEqual(expected, actual)
        self.assertEqual(actual, build_golden())


if __name__ == "__main__":
    if os.getenv("UPDATE_GOLDEN"):
        print(json.dumps(build_golden(), indent=2, sort_keys=True))
    else:
        unittest.main()
