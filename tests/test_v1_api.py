import asyncio
import threading
from dataclasses import replace
from io import BytesIO
import os
from pathlib import Path
import unittest
import zipfile
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import httpx
import numpy as np
from fastapi.testclient import TestClient
from fastapi import HTTPException

from app.api.v1 import AsyncInferenceGate, _driving_field_results, _id_archive, _image_inputs, _run_batch, _safe_entries
from app.artifacts import ArtifactSettings
from app.config import Settings
from app.contracts import DocumentType, ErrorCode
from app.inference.batch import ProfileBatchOutcome, ResourceExhaustedError

# The checked-in Docker example names an image-only model directory.  Keep this
# host test independent of a developer's .env before importing app.main.
os.environ["MODEL_DIR"] = ""
from app.main import create_app
from app.uploads import document_from_bytes

PASSPORT_MRZ = "P<UZBCITIZEN<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<\n0000000000UZB0000000M00000000000000000000000"
ID_MRZ = "I<UZBAD6632763841103741390036<\n7403116F3403277UZB<<<<<<<<<<<8\nEGAMOVA<<IRODA<<<<<<<<<<<<<<<<"


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
        self.job_counts = []

    def run(self, jobs):
        self.calls += 1
        self.job_count = len(jobs)
        self.job_counts.append(len(jobs))
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
            outcomes.append(
                ProfileBatchOutcome(
                    job.item_id,
                    result=({name: None for name in names}, report),
                    mrz_text=PASSPORT_MRZ if job.localization_kind == "mrz" else ID_MRZ if job.mrz_profile else None,
                    mrz_detected=job.localization_kind == "mrz" or job.probe_mrz,
                )
            )
        return outcomes, {"text_detection": {"model_call_count": 1}}


class Models:
    def __init__(self):
        self.runner = Runner()

    def document_aligner(self):
        return lambda **_: np.array([[0, 0], [15, 0], [15, 15], [0, 15]])

    def mrz_scanner(self):
        return lambda image, **_: {"mrz_polygon": [[0, 8], [15, 8], [15, 12], [0, 12]]}

    def profile_batch_runner(self):
        return self.runner

    def preload(self):
        return None

    def readiness(self):
        return {"target": "cpu"}


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

    def test_resource_exhaustion_is_a_request_error_not_a_corrupt_document(self):
        models = Models()
        models.runner.run = lambda _jobs: (_ for _ in ()).throw(ResourceExhaustedError("text detection resource failure"))
        inputs = _image_inputs(DocumentType.PASSPORT, [document_from_bytes(image_bytes(), "passport.jpg")])
        with self.assertRaises(HTTPException) as caught:
            _run_batch(inputs, models, self.settings)
        self.assertEqual(503, caught.exception.status_code)
        self.assertEqual(ErrorCode.RESOURCE_EXHAUSTED, caught.exception.detail["code"])

    def test_all_frozen_routes_are_declared_with_contract_schemas(self):
        schema = create_app(self.settings).openapi()
        paths = schema["paths"]
        for path in (
            "/v1/ocr/passport", "/v1/ocr/id-card", "/v1/ocr/driving-license",
            "/v1/ocr/passport/batch", "/v1/ocr/id-card/batch", "/v1/ocr/driving-license/batch",
            "/v1/health/live", "/v1/health/ready",
        ):
            self.assertIn(path, paths)
        self.assertTrue(all(path.startswith("/v1/") for path in paths if not path.startswith("/verification/")))
        self.assertEqual("#/components/schemas/OcrBatchResponse", paths["/v1/ocr/passport/batch"]["post"]["responses"]["200"]["content"]["application/json"]["schema"]["$ref"])
        request = paths["/v1/ocr/passport/batch"]["post"]["requestBody"]["content"]["multipart/form-data"]["schema"]
        body = schema["components"]["schemas"][request["$ref"].split("/")[-1]]
        self.assertEqual("binary", body["properties"]["images"]["items"]["format"])

    def test_driving_derived_fields_reuse_their_source_evidence(self):
        fields = _driving_field_results(
            {"birth_place": "3. TOSHLOQ", "birth_date": "19.10.2005"},
            {"field_raw_text": {"place_of_birth": ["3. TOSHLOQ 19.10.2005"], "date_of_birth": []}, "field_confidences": {"place_of_birth": None, "date_of_birth": None}, "field_bounding_boxes": {"place_of_birth": None, "date_of_birth": None}},
        )
        self.assertEqual(["3. TOSHLOQ"], fields["birth_place"].raw_text)
        self.assertEqual(["19.10.2005"], fields["birth_date"].raw_text)

    def test_identity_response_includes_mrz(self):
        models = Models()
        inputs = _image_inputs(DocumentType.PASSPORT, [document_from_bytes(image_bytes(), "passport.jpg")])
        response = _run_batch(inputs, models, self.settings)
        self.assertEqual(["P<UZBCITIZEN<<JOHN<<<<<<<<<<<<<<<<<<<<<<<<<<", "0000000000UZB0000000M00000000000000000000000"], response.items[0].result.mrz.raw_lines)

    def test_all_single_and_batch_routes_use_the_same_coordinator(self):
        models = Models()
        with patch("app.main.Models", return_value=models):
            application = create_app(self.settings)
        payload = image_bytes()
        id_zip = archive(
            {
                "one/front.jpg": payload,
                "one/back.jpg": payload,
                "two/front.jpg": payload,
                "two/back.jpg": payload,
            }
        )
        with TestClient(application) as client:
            responses = [
                client.post("/v1/ocr/passport", files={"image": ("p.jpg", payload, "image/jpeg")}),
                client.post("/v1/ocr/passport/batch", files=[("images", ("p1.jpg", payload, "image/jpeg")), ("images", ("p2.jpg", payload, "image/jpeg"))]),
                client.post("/v1/ocr/id-card", files={"front": ("front.jpg", payload, "image/jpeg"), "back": ("back.jpg", payload, "image/jpeg")}),
                client.post("/v1/ocr/id-card/batch", files={"archive": ("ids.zip", id_zip, "application/zip")}),
                client.post("/v1/ocr/driving-license", files={"image": ("d.jpg", payload, "image/jpeg")}),
                client.post("/v1/ocr/driving-license/batch", files=[("images", ("d1.jpg", payload, "image/jpeg")), ("images", ("d2.jpg", payload, "image/jpeg"))]),
            ]
        self.assertEqual([200] * 6, [response.status_code for response in responses])
        self.assertEqual(6, models.runner.calls)
        self.assertEqual([1, 2, 2, 4, 1, 2], models.runner.job_counts)
        self.assertEqual(1, responses[1].json()["diagnostics"]["text_detection"]["model_call_count"])

    def test_v1_writes_request_diagnostics_and_response_artifacts(self):
        with TemporaryDirectory() as directory:
            settings = replace(
                Settings.from_env(),
                artifacts=ArtifactSettings(True, Path(directory)),
            )
            models = Models()
            with patch("app.main.Models", return_value=models):
                application = create_app(settings)
            with TestClient(application) as client:
                response = client.post(
                    "/v1/ocr/passport",
                    files={"image": ("passport.jpg", image_bytes(), "image/jpeg")},
                )

            self.assertEqual(200, response.status_code)
            run = next((Path(directory) / "v1_batch").iterdir())
            self.assertTrue((run / "00_request.json").is_file())
            self.assertTrue((run / "01_pipeline_diagnostics.json").is_file())
            self.assertTrue((run / "02_response.json").is_file())
            self.assertTrue((run / "001_passport").is_dir())


class AsyncInferenceGateTests(unittest.IsolatedAsyncioTestCase):
    async def test_live_health_returns_while_active_inference_runs_off_loop(self):
        class SlowRunner(Runner):
            def __init__(self):
                super().__init__()
                self.started = threading.Event()
                self.release = threading.Event()

            def run(self, jobs):
                self.started.set()
                self.release.wait(2)
                return super().run(jobs)

        base = Settings.from_env()
        settings = replace(base, artifacts=ArtifactSettings(False, base.artifacts.directory))
        models = Models()
        models.runner = SlowRunner()
        with patch("app.main.Models", return_value=models):
            application = create_app(settings)
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=application), base_url="http://test") as client:
            request = asyncio.create_task(
                client.post("/v1/ocr/passport", files={"image": ("p.jpg", image_bytes(), "image/jpeg")})
            )
            self.assertTrue(await asyncio.to_thread(models.runner.started.wait, 2))
            health = await client.get("/v1/health/live")
            self.assertEqual(200, health.status_code)
            self.assertFalse(request.done())
            models.runner.release.set()
            self.assertEqual(200, (await request).status_code)

    async def test_admission_serializes_orders_and_releases_after_errors(self):
        gate = AsyncInferenceGate(3)
        started = threading.Event()
        release = threading.Event()
        order = []

        def first():
            order.append("first")
            started.set()
            release.wait(2)

        first_task = asyncio.create_task(gate.run(first))
        self.assertTrue(await asyncio.to_thread(started.wait, 2))
        second = asyncio.create_task(gate.run(lambda: order.append("second")))
        third = asyncio.create_task(gate.run(lambda: order.append("third")))
        await asyncio.sleep(0)
        self.assertEqual(["first"], order)
        release.set()
        await asyncio.gather(first_task, second, third)
        self.assertEqual(["first", "second", "third"], order)

        with self.assertRaisesRegex(ValueError, "boom"):
            await gate.run(lambda: (_ for _ in ()).throw(ValueError("boom")))
        self.assertEqual("reused", await gate.run(lambda: "reused"))

    async def test_queue_overflow_and_cancelled_waiter_release_their_slots(self):
        gate = AsyncInferenceGate(2)
        started = threading.Event()
        release = threading.Event()

        def slow():
            started.set()
            release.wait(2)

        first = asyncio.create_task(gate.run(slow))
        self.assertTrue(await asyncio.to_thread(started.wait, 2))
        waiting = asyncio.create_task(gate.run(lambda: None))
        await asyncio.sleep(0)
        with self.assertRaisesRegex(Exception, "queue is full"):
            await gate.run(lambda: None)
        waiting.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await waiting
        third = asyncio.create_task(gate.run(lambda: "after-cancel"))
        release.set()
        await first
        self.assertEqual("after-cancel", await third)


if __name__ == "__main__":
    unittest.main()
