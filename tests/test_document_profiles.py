import copy
import json
from pathlib import Path
import tempfile
import unittest

from app.documents.profiles import load_document_profile


ROOT = Path(__file__).resolve().parents[1]
PROFILES = (
    ROOT / "config/documents/uz_passport/profile.json",
    ROOT / "config/documents/uz_id_card/profile.json",
)


class DocumentProfileTests(unittest.TestCase):
    def test_production_profiles_load_without_annotation_truth(self):
        profiles = [load_document_profile(path) for path in PROFILES]
        self.assertEqual([profile["layout"] for profile in profiles], ["uzbekistan_passport", "uzbekistan_id_card"])
        self.assertEqual({field["region"] for field in profiles[1]["fields"]}, {"front", "back"})

    def test_rejects_invalid_bounds_and_duplicate_ownership(self):
        profile = load_document_profile(PROFILES[0])
        bad_bounds = copy.deepcopy(profile)
        bad_bounds["regions"]["data_page"]["field_rois"]["name"]["x2"] = 1.1
        duplicate = copy.deepcopy(profile)
        duplicate["fields"].append(copy.deepcopy(duplicate["fields"][0]))
        missing_size = copy.deepcopy(profile)
        del missing_size["canonical_size"]
        for bad, message in (
            (bad_bounds, "normalized bounds"),
            (duplicate, "duplicate field ownership"),
            (missing_size, "canonical_size"),
        ):
            with self.subTest(message=message), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "profile.json"
                path.write_text(json.dumps(bad), encoding="utf-8")
                with self.assertRaisesRegex(ValueError, message):
                    load_document_profile(path)
