import json
import os
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import cv2
from fastapi.testclient import TestClient

from app.artifacts import ArtifactSettings
from app.config import Settings
from app.inference.batch import OcrBatchResult
from app.main import create_app
from app.verification import VerificationLine, verify_fields


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
ANNOTATIONS = DATASET / "annotations"
HAS_DATASET = ANNOTATIONS.is_dir()


def _annotations():
    return [json.loads(path.read_text()) for path in sorted(ANNOTATIONS.glob("*/*.json"))]


def _values(annotation):
    return {
        name: field["value"]
        for name, field in annotation["fields"].items()
        if field.get("state") == "value" and field.get("value")
    }


@unittest.skipUnless(HAS_DATASET, "local dataset/ annotations are unavailable")
class VerificationDatasetTests(unittest.TestCase):
    def test_all_annotated_images_and_unknown_field_names_are_accepted(self):
        annotations = _annotations()
        self.assertEqual(20, len(annotations))
        for annotation in annotations:
            with self.subTest(document=annotation["id"]):
                for relative_path in annotation["images"].values():
                    image = cv2.imread(str(DATASET / relative_path))
                    self.assertIsNotNone(image, relative_path)
                    self.assertGreater(image.size, 0)
                values = _values(annotation)
                result = verify_fields(
                    [VerificationLine(value, 0.9) for value in values.values()],
                    values,
                )
                self.assertEqual(len(values), result["summary"]["match"])

    def test_representative_dataset_images_pass_all_ocr_transport_shapes(self):
        class FakeOcr:
            def run(self, samples):
                return OcrBatchResult(
                    tokens={sample.item_id: [{"text": "fixture-line", "score": 0.9}] for sample in samples},
                    errors={},
                    diagnostics={},
                )

        class FakeModels:
            def verification_ocr(self):
                return FakeOcr()

            def preload(self):
                pass

            def close(self):
                pass

        by_type = {
            "passport": next(item for item in _annotations() if item["document_type"] == "passport"),
            "id_card": next(item for item in _annotations() if item["document_type"] == "id_card"),
            "driving_license": next(item for item in _annotations() if item["document_type"] == "driving_license"),
        }
        settings = replace(Settings.from_env(), artifacts=ArtifactSettings(False, Settings.from_env().artifacts.directory))
        with patch("app.main.Models", return_value=FakeModels()):
            application = create_app(settings)
        with TestClient(application) as client:
            passport = by_type["passport"]
            passport_path = DATASET / passport["images"]["image"]
            response = client.post(
                "/verification/passport/ocr",
                files={"image": (passport_path.name, passport_path.read_bytes(), "image/png")},
            )
            self.assertEqual(200, response.status_code)
            self.assertTrue(response.json()["lines"])

            card = by_type["id_card"]
            card_files = {
                side: (Path(relative).name, (DATASET / relative).read_bytes(), "image/png")
                for side, relative in card["images"].items()
            }
            response = client.post("/verification/id-card/ocr", files=card_files)
            self.assertEqual(200, response.status_code)
            self.assertEqual({"front", "back"}, set(response.json()))

            licence = by_type["driving_license"]
            licence_path = DATASET / licence["images"]["image"]
            response = client.post(
                "/verification/driving-licence/ocr",
                files={"image": (licence_path.name, licence_path.read_bytes(), "image/jpeg")},
            )
            self.assertEqual(200, response.status_code)
            self.assertTrue(response.json()["lines"])
REAL_CACHE = Path(os.getenv("VOIGHT_REAL_MODEL_CACHE", ""))
HAS_REAL_CACHE = bool(os.getenv("VOIGHT_RUN_REAL_E2E")) and REAL_CACHE.is_dir() and all(
    (REAL_CACHE / "official_models" / name / "inference.pdiparams").is_file()
    for name in ("PP-OCRv6_medium_det", "latin_PP-OCRv5_mobile_rec")
)


@unittest.skipUnless(HAS_DATASET and HAS_REAL_CACHE, "opt-in CPU model cache or dataset unavailable")
class RealVerificationDatasetTests(unittest.TestCase):
    def test_real_cpu_routes_and_checks_use_annotated_dataset_fixtures(self):
        with patch.dict(
            os.environ,
            {
                "RUNTIME_TARGET": "cpu",
                "OCR_DEVICE": "cpu",
                "MODEL_DIR": str(REAL_CACHE),
                "PRELOAD": "false",
                "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "true",
            },
            clear=False,
        ):
            application = create_app(replace(Settings.from_env(), artifacts=ArtifactSettings(False, ROOT / "logs")))
        fixtures = {
            "passport": ("/verification/passport/ocr", "/verification/passport/check", "p_1.json"),
            "id_card": ("/verification/id-card/ocr", "/verification/id-card/check", "id_1.json"),
            "driving_license": ("/verification/driving-licence/ocr", "/verification/driving-licence/check", "d_1.json"),
        }
        with TestClient(application) as client:
            for kind, (ocr_route, check_route, filename) in fixtures.items():
                annotation = json.loads(next((ANNOTATIONS / kind).glob(filename)).read_text())
                files = {
                    side: (Path(relative).name, (DATASET / relative).read_bytes(), "image/png")
                    for side, relative in annotation["images"].items()
                }
                if kind != "id_card":
                    files = {"image": next(iter(files.values()))}
                ocr = client.post(ocr_route, files=files)
                self.assertEqual(200, ocr.status_code, kind)
                payload = ocr.json()
                self.assertTrue(payload.get("lines") or payload.get("front") or payload.get("back"))
                check = client.post(check_route, json={"ocr": payload, "fields": _values(annotation)})
                self.assertEqual(200, check.status_code, kind)
                self.assertEqual(set(_values(annotation)), set(check.json()["fields"]))


if __name__ == "__main__":
    unittest.main()
