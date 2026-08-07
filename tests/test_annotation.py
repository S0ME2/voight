import json
from pathlib import Path
import subprocess
import sys
import unittest

import cv2
import numpy as np

from app.annotation import AnnotationStore, Sample, discover_inputs, validate_corners, validate_rect, write_outputs


def image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), np.zeros((20, 30, 3), dtype=np.uint8))


class AnnotationTests(unittest.TestCase):
    def test_discovery_pairing_and_order(self):
        with self.subTest("valid"):
            from tempfile import TemporaryDirectory
            with TemporaryDirectory() as directory:
                root = Path(directory)
                image(root / "passports" / "B.JPG")
                image(root / "passports" / "a.png")
                image(root / "id_cards" / "z" / "front.jpg")
                image(root / "id_cards" / "z" / "back.jpg")
                self.assertEqual([sample.key for sample in discover_inputs(root)], ["passport:a.png", "passport:B.JPG", "id_card:z:front", "id_card:z:back"])

    def test_discovery_rejects_missing_or_duplicate_id_side(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image(root / "id_cards" / "missing" / "front.jpg")
            with self.assertRaisesRegex(ValueError, "missing back"):
                discover_inputs(root)
            image(root / "id_cards" / "missing" / "back.jpg")
            image(root / "id_cards" / "missing" / "front.png")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                discover_inputs(root)

    def test_normalized_validation(self):
        self.assertEqual(validate_rect({"x1": 0, "y1": 0, "x2": 1, "y2": 1})["x2"], 1)
        with self.assertRaises(ValueError):
            validate_rect({"x1": 0.8, "y1": 0, "x2": 0.2, "y2": 1})
        self.assertEqual(len(validate_corners([[0, 0], [1, 0], [1, 1], [0, 1]])), 4)
        with self.assertRaises(ValueError):
            validate_corners([[0, 0], [1, 1], [1, 0], [0, 1]])

    def test_atomic_resume_profiles_and_check_report(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "passports" / "one.jpg"
            image(source)
            sample = Sample("passport:one.jpg", "passport", "uzbekistan_passport", None, source)
            store = AnnotationStore(root / "output", [sample])
            saved = json.loads(store.path.read_text())
            self.assertEqual(saved["samples"][sample.key]["original_size"], {"width": 30, "height": 20})
            item = store.sample(sample.key)
            item.update({"status": "complete", "corners": [[0, 0], [1, 0], [1, 1], [0, 1]], "data_crop": {"x1": 0, "y1": 0, "x2": 1, "y2": 1}, "fields": {"number": {"x1": 0.1, "y1": 0.1, "x2": 0.5, "y2": 0.2}}, "expected_fields": {"number": "AA1"}, "mrz": "P<"})
            store.save()
            report = write_outputs(AnnotationStore(root / "output", [sample]))
            self.assertTrue(report["valid"])
            self.assertEqual(json.loads((root / "output" / "profiles" / "uzbekistan_passport.json").read_text())["field_rois"]["number"]["x2"], 0.5)
            self.assertEqual(json.loads((root / "output" / "evaluation_ground_truth.json").read_text())["samples"][sample.key]["expected_fields"]["number"], "AA1")
            checked = subprocess.run([sys.executable, "scripts/annotate.py", str(root), str(root / "output"), "--check"], capture_output=True, text=True, check=False)
            self.assertEqual(checked.returncode, 0, checked.stderr)
            self.assertIn('"valid": true', checked.stdout)

    def test_resume_accepts_newly_added_input(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image(root / "id_cards" / "one" / "front.jpg")
            image(root / "id_cards" / "one" / "back.jpg")
            store = AnnotationStore(root / "output", discover_inputs(root))
            image(root / "passports" / "one.jpg")
            resumed = AnnotationStore(root / "output", discover_inputs(root))
            self.assertEqual(set(resumed.data["samples"]), {item.key for item in discover_inputs(root)})
            self.assertEqual(store.path, resumed.path)

    def test_json_layout_defines_driving_license_annotation(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image(root / "driving_licenses" / "one.jpg")
            layouts = root / "layouts.json"
            layouts.write_text(json.dumps({"layouts": [{
                "input_directory": "driving_licenses", "document_type": "driving_license", "layout": "driving_license",
                "annotation_mode": "canonical", "canonical_size": {"width": 1000, "height": 630}, "fields": ["license_number"]
            }]}), encoding="utf-8")
            sample = discover_inputs(root, layouts)[0]
            self.assertEqual(sample.annotation_mode, "canonical")
            store = AnnotationStore(root / "output", [sample])
            self.assertEqual(store.sample(sample.key)["coordinate_space"], "canonical")

    def test_check_rejects_malformed_annotation(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "passports" / "one.jpg"
            image(source)
            sample = Sample("passport:one.jpg", "passport", "uzbekistan_passport", None, source)
            store = AnnotationStore(root / "output", [sample])
            store.sample(sample.key).update({"status": "complete", "corners": [[0, 0]]})
            self.assertFalse(write_outputs(store)["valid"])
