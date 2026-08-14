import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.dataset.annotate import (
    FIELDS,
    MRZ_SHAPES,
    Document,
    annotate_one,
    annotation_path,
    atomic_json,
    dataset_summary,
    discover_documents,
    export_jsonl,
    mrz_warnings,
    new_annotation,
    run_annotation,
    sha256,
    validate_dataset,
)


def image(path: Path, content: bytes = b"unchanged-image-bytes") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def complete_annotation(root: Path, document: Document) -> dict:
    data = new_annotation(root, document)
    data["fields"] = {
        name: {"state": "value", "value": f"ground-truth-{name}"}
        for name in FIELDS[document.document_type]
    }
    if document.document_type in MRZ_SHAPES:
        count, width = MRZ_SHAPES[document.document_type]
        data["mrz"] = {"lines": ["<" * width for _ in range(count)]}
    data.update(notes="", status="complete")
    atomic_json(annotation_path(root, document), data)
    return data


def finishing_inputs(document_type: str, *, skip_fields: int = 0) -> list[str]:
    values = [f"value-{name}" for name in FIELDS[document_type][skip_fields:]]
    if document_type in MRZ_SHAPES:
        count, width = MRZ_SHAPES[document_type]
        values.extend("<" * width for _ in range(count))
    return values + [""]


class DatasetAnnotationTests(unittest.TestCase):
    def test_discovers_passport_images_and_ignores_unsupported_files(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image(root / "passport" / "passport_002.PNG")
            image(root / "passport" / "passport_001.jpg")
            image(root / "passport" / "ignore.gif")
            image(root / "passport" / "notes.txt")
            documents = discover_documents(root)
            self.assertEqual([item.id for item in documents], ["passport_001", "passport_002"])

    def test_discovers_id_card_front_back_as_one_document(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image(root / "id_card" / "card_001" / "front.webp")
            image(root / "id_card" / "card_001" / "back.jpeg")
            documents = discover_documents(root)
            self.assertEqual(len(documents), 1)
            self.assertEqual(documents[0].images, {
                "front": "id_card/card_001/front.webp",
                "back": "id_card/card_001/back.jpeg",
            })

    def test_incomplete_id_card_pair_is_rejected(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image(root / "id_card" / "card_001" / "front.jpg")
            with self.assertRaisesRegex(ValueError, "missing back"):
                discover_documents(root)

    def test_discovers_driving_licences(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image(root / "driving_license" / "licence_001.webp")
            document = discover_documents(root)[0]
            self.assertEqual((document.document_type, document.id), ("driving_license", "licence_001"))

    def test_creates_canonical_annotation_json(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = image(root / "passport" / "passport_001.jpg")
            document = discover_documents(root)[0]
            responses = iter(finishing_inputs("passport"))
            self.assertEqual(annotate_one(root, document, input_fn=lambda _: next(responses)), "complete")
            data = json.loads(annotation_path(root, document).read_text())
            self.assertEqual(data["id"], "passport_001")
            self.assertEqual(data["images"], {"image": "passport/passport_001.jpg"})
            self.assertEqual(data["image_sha256"]["image"], sha256(source))
            self.assertEqual(set(data["fields"]), set(FIELDS["passport"]))
            self.assertEqual(data["status"], "complete")

    def test_resume_preserves_existing_field_and_starts_at_first_missing(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image(root / "passport" / "one.jpg")
            document = discover_documents(root)[0]
            data = new_annotation(root, document)
            first = FIELDS["passport"][0]
            data["fields"][first] = {"state": "value", "value": "KEEP ME"}
            atomic_json(annotation_path(root, document), data)
            prompts = []
            responses = iter(finishing_inputs("passport", skip_fields=1))
            annotate_one(root, document, input_fn=lambda prompt: prompts.append(prompt) or next(responses))
            resumed = json.loads(annotation_path(root, document).read_text())
            self.assertEqual(resumed["fields"][first]["value"], "KEEP ME")
            self.assertIn(FIELDS["passport"][1], prompts[0])

    def test_completed_document_is_skipped_by_default(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image(root / "driving_license" / "one.jpg")
            document = discover_documents(root)[0]
            complete_annotation(root, document)
            messages = []
            result = run_annotation(
                root, [document], output=messages.append,
                viewer=lambda *_: self.fail("viewer must not open for a completed document"),
            )
            self.assertEqual(result, 0)
            self.assertIn("No matching unfinished documents.", messages)

    def test_viewer_keeps_control_of_terminal_input(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image(root / "driving_license" / "one.jpg")
            document = discover_documents(root)[0]

            class Viewer:
                def __init__(self):
                    self.responses = iter(finishing_inputs("driving_license"))
                    self.closed = False

                def read_input(self, _prompt):
                    return next(self.responses)

                def __call__(self):
                    self.closed = True

            viewer = Viewer()
            self.assertEqual(run_annotation(root, [document], viewer=lambda *_: viewer), 0)
            self.assertTrue(viewer.closed)
            self.assertEqual(json.loads(annotation_path(root, document).read_text())["status"], "complete")

    def test_quit_preserves_entered_progress(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image(root / "passport" / "one.jpg")
            document = discover_documents(root)[0]
            responses = iter(["MANUAL VALUE", ":q"])
            self.assertEqual(annotate_one(root, document, input_fn=lambda _: next(responses)), "quit")
            data = json.loads(annotation_path(root, document).read_text())
            self.assertEqual(data["fields"][FIELDS["passport"][0]], {"state": "value", "value": "MANUAL VALUE"})
            self.assertEqual(data["status"], "in_progress")

    def test_sha256_is_stable_and_validation_detects_change_and_missing_image(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = image(root / "driving_license" / "one.jpg")
            document = discover_documents(root)[0]
            original = sha256(source)
            self.assertEqual(original, sha256(source))
            complete_annotation(root, document)
            source.write_bytes(b"changed")
            self.assertTrue(any("changed SHA-256" in issue for issue in validate_dataset(root)))
            source.unlink()
            self.assertTrue(any("missing image" in issue for issue in validate_dataset(root)))

    def test_mrz_validation_warns_without_changing_text(self):
        line = "Abdulaev<123"
        warnings = mrz_warnings("passport", [line, "<" * 44])
        self.assertTrue(any("expected 44" in warning for warning in warnings))
        self.assertTrue(any("outside A-Z" in warning for warning in warnings))
        self.assertEqual(line, "Abdulaev<123")

    def test_visible_fields_and_mrz_may_disagree(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image(root / "passport" / "one.jpg")
            document = discover_documents(root)[0]
            data = complete_annotation(root, document)
            data["fields"]["surname"] = {"state": "value", "value": "ABDULLAYEV"}
            data["mrz"]["lines"][0] = "P<UZBABDULAEV<<AKMAL<<<<<<<<<<<<<<<<<<<<<<"
            atomic_json(annotation_path(root, document), data)
            saved = json.loads(annotation_path(root, document).read_text())
            self.assertEqual(saved["fields"]["surname"]["value"], "ABDULLAYEV")
            self.assertIn("ABDULAEV", saved["mrz"]["lines"][0])

    def test_export_generates_valid_jsonl(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image(root / "passport" / "p.jpg")
            image(root / "driving_license" / "d.png")
            for document in discover_documents(root):
                complete_annotation(root, document)
            exported = export_jsonl(root)
            objects = [json.loads(line) for line in exported.read_text().splitlines()]
            self.assertEqual({item["id"] for item in objects}, {"p", "d"})

    def test_annotation_never_modifies_original_image(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            source = image(root / "driving_license" / "one.jpg", b"original-exact-bytes")
            before = source.read_bytes()
            document = discover_documents(root)[0]
            responses = iter(finishing_inputs("driving_license"))
            annotate_one(root, document, input_fn=lambda _: next(responses))
            self.assertEqual(source.read_bytes(), before)

    def test_field_states_are_distinct(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image(root / "driving_license" / "one.jpg")
            document = discover_documents(root)[0]
            responses = iter(["", ":empty", ":unreadable", ":q"])
            annotate_one(root, document, input_fn=lambda _: next(responses))
            fields = json.loads(annotation_path(root, document).read_text())["fields"]
            names = FIELDS["driving_license"]
            self.assertEqual(fields[names[0]], {"state": "value", "value": ""})
            self.assertEqual(fields[names[1]], {"state": "empty", "value": None})
            self.assertEqual(fields[names[2]], {"state": "unreadable", "value": None})
            self.assertNotIn(names[3], fields)

    def test_validation_reports_unknown_and_missing_fields(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image(root / "driving_license" / "one.jpg")
            document = discover_documents(root)[0]
            data = complete_annotation(root, document)
            data["fields"].pop(FIELDS["driving_license"][0])
            data["fields"]["invented_name"] = {"state": "value", "value": "x"}
            atomic_json(annotation_path(root, document), data)
            issues = validate_dataset(root)
            self.assertTrue(any("unknown fields" in issue for issue in issues))
            self.assertTrue(any("missing annotation fields" in issue for issue in issues))

    def test_summary_counts_status_and_fields(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            image(root / "passport" / "one.jpg")
            document = discover_documents(root)[0]
            complete_annotation(root, document)
            summary = dataset_summary(root, [document])
            self.assertIn("complete: 1", summary)
            self.assertIn(f"fields annotated: {len(FIELDS['passport'])}", summary)


if __name__ == "__main__":
    unittest.main()
