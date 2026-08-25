import os
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


ROOT = Path(__file__).resolve().parents[1]
CACHE_ROOT = Path(os.getenv("VOIGHT_REAL_MODEL_CACHE", ""))
CACHE = bool(os.getenv("VOIGHT_RUN_REAL_E2E")) and CACHE_ROOT.is_dir() and all(
    (CACHE_ROOT / "official_models" / name / "inference.pdiparams").is_file()
    for name in ("PP-OCRv6_medium_det", "latin_PP-OCRv5_mobile_rec")
)


@unittest.skipUnless(CACHE, "complete local model cache not selected; no download is allowed")
class RealV1FixtureTests(unittest.TestCase):
    def test_one_v1_request_per_committed_document_fixture(self):
        with mock.patch.dict(os.environ, {"MODEL_DIR": str(CACHE_ROOT), "PRELOAD": "false"}, clear=False):
            application = create_app(Settings.from_env())
        fixtures = {
            "passport": (
                "/v1/ocr/passport",
                {"image": ("passport.png", (ROOT / "annotation_input/passports/passport.png").read_bytes(), "image/png")},
                {"type": "P", "country_code": "UZB", "passport_number": "000000000", "surname": "CITIZEN", "name": "JOHN", "patronymic": "DOE", "nationality": "UZBEKISTAN", "date_of_birth": "01.01.2000", "sex": "M", "place_of_birth": "TASHKENT", "date_of_issue": "01.01.2020", "date_of_expiry": "01.01.2030", "authority": "IIV"},
                ["P<UZBCITIZEN<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<", "0000000000UZB0000000M00000000000000000000000"],
            ),
            "id_card": (
                "/v1/ocr/id-card",
                {"front": ("front.png", (ROOT / "annotation_input/id_cards/uzbekistan_id_001/front.png").read_bytes(), "image/png"), "back": ("back.png", (ROOT / "annotation_input/id_cards/uzbekistan_id_001/back.png").read_bytes(), "image/png")},
                {"surname": "EGAMOVA", "name": "IRODA", "patronymic": "IBROXIMOVNA", "date_of_birth": "11.03.1974", "date_of_issue": "28.03.2024", "date_of_expiry": "27.03.2034", "sex": "AYOL", "citizenship": "O'ZBEKISTON", "card_number": "AD6632763", "pinfl": "41103741390036", "place_of_birth": "SHAXRIXON", "place_of_issue": "IIV3234"},
                ["I<UZBAD6632763841103741390036<", "7403116F3403277UZB<<<<<<<<<<<8", "EGAMOVA<<IRODA<<<<<<<<<<<<<<<<"],
            ),
            "driving_license": (
                "/v1/ocr/driving-license",
                {"image": ("license.jpg", (ROOT / "annotation_input/driving_licenses/test_license_canonical.jpg").read_bytes(), "image/jpeg")},
                {},
                [],
            ),
        }
        with TestClient(application) as client:
            for name, (route, files, fields, lines) in fixtures.items():
                with self.subTest(document=name):
                    response = client.post(route, files=files)
                    self.assertEqual(200, response.status_code)
                    result = response.json()["result"]
                    for field, value in fields.items():
                        self.assertEqual(value, result["fields"][field]["value"], field)
                    if lines:
                        self.assertEqual(lines, result["mrz"]["raw_lines"])


if __name__ == "__main__":
    unittest.main()
