import asyncio
import hashlib
import json
import tempfile
import unittest
import zipfile
from dataclasses import replace
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from app.api.v1 import AsyncInferenceGate, _id_archive, _safe_entries
from app.artifacts import (
    ArtifactSettings,
    ArtifactWriter,
    json_default,
    new_run_id,
    next_counter,
    safe_filename,
)
from app.config import ModelSettings, ProfileSettings, Settings
from app.contracts import DocumentType, ValidationStatus
from app.documents.mrz import ID_CARD, PASSPORT, check_digit, parse
from app.inference.batch import BatchedOcr, OcrSample
from app.inference.contracts import DetectedTextRegion, DetectedTextRegions, RecognitionResult
from app.inference.packing import recognition_batch_packer
from app.uploads import document_from_bytes


ROOT = Path(__file__).resolve().parents[1]


def image_bytes() -> bytes:
    ok, encoded = cv2.imencode(".jpg", np.zeros((16, 16, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def archive(entries: dict[str, bytes], encrypted: bool = False) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w") as zipped:
        for name, data in entries.items():
            info = zipfile.ZipInfo(name)
            info.flag_bits = 1 if encrypted else 0
            zipped.writestr(info, data)
    return output.getvalue()


class ConfigMatrixTests(unittest.TestCase):
    def setUp(self):
        with patch.dict("os.environ", {}, clear=True):
            self.settings = Settings.from_env()

    def test_validate_startup_rejects_every_typed_boundary(self):
        cases = [
            (replace(self.settings, runtime=replace(self.settings.runtime, target="bad")), "RUNTIME_TARGET"),
            (replace(self.settings, ocr=replace(self.settings.ocr, device="bad")), "OCR_DEVICE"),
            (replace(self.settings, runtime=replace(self.settings.runtime, target="gpu")), "same runtime"),
            (replace(self.settings, runtime=replace(self.settings.runtime, gpu_id=-1)), "GPU_ID"),
            (replace(self.settings, runtime=replace(self.settings.runtime, text_recognition_processes=2, target="gpu"), ocr=replace(self.settings.ocr, device="gpu")), "only on CPU"),
            (replace(self.settings, runtime=replace(self.settings.runtime, text_recognition_precision="int8")), "PRECISION"),
            (replace(self.settings, runtime=replace(self.settings.runtime, text_recognition_packing="bad")), "PACKING"),
            (replace(self.settings, driving_license=replace(self.settings.driving_license, aligner_model_type="bad")), "MODEL_TYPE"),
            (replace(self.settings, runtime=replace(self.settings.runtime, text_recognition_enable_hpi=True)), "require RUNTIME_TARGET"),
            (replace(self.settings, runtime=replace(self.settings.runtime, text_recognition_use_tensorrt=True)), "require RUNTIME_TARGET"),
            (replace(self.settings, runtime=replace(self.settings.runtime, text_recognition_precision="fp16")), "require RUNTIME_TARGET"),
            (replace(self.settings, profiles=ProfileSettings(Path("/missing"), self.settings.profiles.id_card)), "PASSPORT_PROFILE"),
            (replace(self.settings, models=ModelSettings(Path("/missing"))), "MODEL_DIR"),
            (replace(self.settings, batch=replace(self.settings.batch, max_archive_uncompressed_bytes=1)), "at least"),
        ]
        for invalid, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    invalid.validate_startup()

        positive = (
            "cpu_threads", "queue_limit", "localization_batch_size",
            "text_detection_batch_size", "text_recognition_batch_size",
            "mrz_recognition_batch_size", "text_recognition_processes",
            "text_detector_pixel_scale", "text_detector_limit_side_len",
        )
        for name in positive:
            value = replace(self.settings.runtime, **{name: 0})
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name.upper()):
                replace(self.settings, runtime=value).validate_startup()

    def test_blank_model_selection_is_rejected(self):
        invalid = replace(
            self.settings,
            models=replace(self.settings.models, text_detector=replace(self.settings.models.text_detector, model=" ")),
        )
        with self.assertRaisesRegex(ValueError, "TEXT_DETECTOR_MODEL"):
            invalid.validate_startup()


class MrzMatrixTests(unittest.TestCase):
    passport = "P<UZBCITIZEN<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<\n0000000000UZB0000000M00000000000000000000000"
    id_card = "I<UZBAD6632763841103741390036<\n7403116F3403277UZB<<<<<<<<<<<8\nEGAMOVA<<IRODA<<<<<<<<<<<<<<<<"

    def test_td3_td1_golden_corpus_and_composite_digits(self):
        for text, kind, expected_lines in ((self.passport, "passport", 2), (self.id_card, "id_card", 3)):
            with self.subTest(kind=kind):
                result = parse(text, kind)
                self.assertEqual(expected_lines, len(result.raw_lines))
                self.assertTrue(all(item.status is ValidationStatus.PASSED for item in result.validations))
                self.assertEqual("6", check_digit("740311"))

    def test_every_check_digit_failure_is_reported(self):
        for kind, source in (("passport", self.passport), ("id_card", self.id_card)):
            lines = source.splitlines()
            indexes = (9, 19, 27, 42, 43) if kind == "passport" else (14, 6, 14, 29)
            line_index = 1
            for index in indexes:
                mutated = list(lines[line_index])
                mutated[index] = "1" if mutated[index] != "1" else "2"
                altered = lines.copy()
                altered[line_index] = "".join(mutated)
                result = parse("\n".join(altered), kind)
                with self.subTest(kind=kind, index=index):
                    self.assertIn(ValidationStatus.FAILED, [item.status for item in result.validations])

    def test_split_concatenated_lines_garbage_and_wrong_type(self):
        for text, kind, width, count in ((self.passport.replace("\n", ""), "passport", 44, 2), (self.id_card.replace("\n", ""), "id_card", 30, 3)):
            result = parse(text, kind)
            self.assertEqual(count, len(result.raw_lines))
            self.assertEqual(width, len(result.raw_lines[0]))
        self.assertEqual(ValidationStatus.FAILED, parse("garbage", "passport").validations[0].status)
        with self.assertRaisesRegex(ValueError, "unsupported"):
            parse(self.passport, "driving_license")


class PackingMatrixTests(unittest.TestCase):
    def test_all_packers_restore_results_and_fixed_width_isolation(self):
        class Detector:
            def detect_batch(self, images):
                return [DetectedTextRegions((DetectedTextRegion(np.asarray([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)),)) for h, w in (image.shape[:2] for image in images)]

        class Recognizer:
            def recognize_batch(self, images):
                return [RecognitionResult(str(int(image[0, 0, 0])), 1.0) for image in images]

        shapes = ((24, 80), (32, 40), (20, 160), (28, 60), (40, 100))
        for name in ("sequential", "aspect-ratio", "fixed-width", "fixed-width-buckets", "best-fit"):
            with self.subTest(name=name):
                result = BatchedOcr(Detector(), Recognizer(), detection_batch_size=5, recognition_batch_size=2, recognition_packer=recognition_batch_packer(name)).run(
                    [OcrSample(str(index), np.full((*shape, 3), index + 1, np.uint8)) for index, shape in enumerate(shapes)]
                )
                self.assertEqual([str(index + 1) for index in range(len(shapes))], [result.tokens[str(index)][0]["text"] for index in range(len(shapes))])

    def test_best_fit_rejects_any_batch_size_other_than_two(self):
        with self.assertRaisesRegex(ValueError, "batch=2"):
            recognition_batch_packer("best-fit").pack([(0, np.zeros((10, 10), np.uint8))], 3)

    def test_fixed_width_prepared_tensor_hash_does_not_depend_on_neighbors(self):
        packer = recognition_batch_packer("fixed-width")
        image = np.full((20, 60, 3), 80, np.uint8)
        solo = packer.pack([(0, image)], 1)[0][0][1]
        pair = packer.pack([(0, image), (1, np.full_like(image, 160))], 2)[0][0][1]
        self.assertEqual(hashlib.sha256(solo.tobytes()).hexdigest(), hashlib.sha256(pair.tobytes()).hexdigest())


class ZipSecurityMatrixTests(unittest.TestCase):
    def setUp(self):
        self.settings = Settings.from_env()
        self.archive_document = lambda data: document_from_bytes(data, "input.zip")

    def assertArchiveError(self, data, message, settings=None):
        with self.assertRaisesRegex(Exception, message):
            _safe_entries(self.archive_document(data), settings or self.settings)

    def test_paths_encryption_sizes_count_and_id_pairing(self):
        self.assertArchiveError(archive({"/front.jpg": image_bytes()}), "unsafe path")
        self.assertArchiveError(archive({"../front.jpg": image_bytes()}), "unsafe path")
        self.assertArchiveError(archive({"card/front.jpg": b"x" * 21}), "BATCH_MAX_ARCHIVE", replace(self.settings, batch=replace(self.settings.batch, max_archive_uncompressed_bytes=20)))
        self.assertArchiveError(archive({"card/front.jpg": b"x" * (self.settings.batch.max_file_bytes + 1)}), "BATCH_MAX_FILE_BYTES")
        self.assertArchiveError(archive({f"card/{index}.jpg": image_bytes() for index in range(3)}), "too many", replace(self.settings, batch=replace(self.settings.batch, max_files=1)))
        encrypted = archive({"card/front.jpg": image_bytes()}, encrypted=True)
        with patch("app.api.v1.zipfile.ZipFile") as zipped:
            class Entry:
                filename, file_size, flag_bits = "card/front.jpg", 1, 1
                def is_dir(self): return False
            zipped.return_value.__enter__.return_value.infolist.return_value = [Entry()]
            self.assertArchiveError(encrypted, "Encrypted ZIP")
        with self.assertRaisesRegex(Exception, "requires front and back"):
            _id_archive(self.archive_document(archive({"card/front.jpg": image_bytes()})), self.settings)
        with self.assertRaisesRegex(Exception, "duplicate front"):
            _id_archive(self.archive_document(archive({"card/front.jpg": image_bytes(), "card/front.png": image_bytes(), "card/back.jpg": image_bytes()})), self.settings)


class ArtifactAndWorkerTests(unittest.TestCase):
    def test_artifact_names_counters_disabled_writes_and_json_default(self):
        self.assertEqual("file", safe_filename("../../"))
        self.assertEqual("a-b", safe_filename("a b.txt"))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "runs"
            (root / "op" / "2_old").mkdir(parents=True)
            (root / "op" / "7_new").mkdir()
            self.assertEqual(8, next_counter(root / "op"))
            run = new_run_id(root, "op", "../a b.jpg", False)
            self.assertTrue(run.startswith("8_a-b_"))
            disabled = ArtifactWriter(root, "op", "1_test", False)
            disabled.save_text("x.txt", "no")
            self.assertFalse((disabled.directory / "x.txt").exists())
            enabled = ArtifactWriter(root, "op", "9_test", True)
            enabled.directory.mkdir(parents=True)
            enabled.save_json("data.json", {"array": np.array([1, 2]), "path": Path("x"), "set": {"a"}})
            self.assertEqual({"array": [1, 2], "path": "x", "set": ["a"]}, json.loads(enabled.path("data.json").read_text()))
        self.assertEqual([1, 2], json_default(np.array([1, 2])))

    def test_process_recognizer_adapter_lifecycle_and_chunk_round_trip(self):
        from app.inference.paddle import ProcessTextRecognizer

        class Worker:
            model_name = "stub"
            def __init__(self): self.calls = []; self.started = 0; self.closed = 0
            def start(self): self.started += 1
            def close(self): self.closed += 1
            def predict_chunks(self, chunks):
                self.calls.append(chunks)
                return [([{"rec_text": f"v{len(chunk)}", "rec_score": 0.9} for _ in chunk], 0.01) for chunk in chunks]

        worker = Worker()
        adapter = ProcessTextRecognizer(worker)
        adapter.start()
        values = adapter.recognize_batch([np.zeros((4, 4, 3), np.uint8)] * 2)
        adapter.close()
        self.assertEqual(["v2", "v2"], [value.text for value in values])
        self.assertEqual((2, 0.9), (len(worker.calls[0][0]), values[0].score))
        self.assertEqual((1, 1), (worker.started, worker.closed))


class GateMatrixTests(unittest.IsolatedAsyncioTestCase):
    async def test_admitted_limit_maps_to_queue_full_and_releases_on_failure(self):
        gate = AsyncInferenceGate(1)
        claim = gate.claim()
        await claim.__aenter__()
        with self.assertRaisesRegex(Exception, "queue is full"):
            await gate.claim().__aenter__()
        await claim.__aexit__(None, None, None)
        with self.assertRaisesRegex(ValueError, "boom"):
            await gate.run(lambda: (_ for _ in ()).throw(ValueError("boom")))
        self.assertEqual("ok", await gate.run(lambda: "ok"))


if __name__ == "__main__":
    unittest.main()
