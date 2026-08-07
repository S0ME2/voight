import json
import tempfile
import unittest
from pathlib import Path

from app.documents.profiles import load_document_profile
from scripts.promote_annotations import promote


ROOT = Path(__file__).resolve().parents[1]


class AnnotationPromotionTests(unittest.TestCase):
    def test_promotes_current_annotations_and_preserves_passport_sex_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "documents"
            (destination / "uz_passport").mkdir(parents=True)
            (destination / "uz_id_card").mkdir()
            for profile in ("uz_passport", "uz_id_card"):
                source = ROOT / "config/documents" / profile / "profile.json"
                (destination / profile / "profile.json").write_text(source.read_text(), encoding="utf-8")
            promote(ROOT / "annotations/annotation_state.json", destination)
            passport = load_document_profile(destination / "uz_passport/profile.json")
            self.assertIn("sex", passport["regions"]["data_page"]["field_rois"])
            self.assertNotIn("sec", passport["regions"]["data_page"]["field_rois"])
            self.assertIn("nationality", passport["regions"]["data_page"]["field_rois"])
