"""Frozen v1 HTTP transport, kept separate from migration-era routes."""

from __future__ import annotations

import asyncio
import os
import time
import zipfile
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import PurePosixPath

from fastapi import APIRouter, File, HTTPException, UploadFile
import anyio

from app.api.schemas import OcrBatchResponse, OcrResponse
from app.api.upload_types import BatchUploads, OptionalUpload
from app.api.v1_batch import (
    LogicalInput,
    assemble_batch_response,
    build_batch_plan,
    driving_field_results as _driving_field_results,
)
from app.config import Settings
from app.artifacts import create_batch_artifact_run
from app.contracts import (
    DocumentInput,
    DocumentType,
    ErrorCode,
    ErrorResult,
)
from app.inference import QueueFullError, ResourceExhaustedError
from app.models import Models
from app.uploads import UploadedDocument, document_from_bytes, image_from_document


class AsyncInferenceGate:
    """Own `/v1` admission: at most ``limit`` admitted requests, one executing."""

    def __init__(self, limit: int):
        self.limit = limit
        self._admitted = 0
        self._admission_lock = asyncio.Lock()
        self._execution_lock = asyncio.Lock()

    @asynccontextmanager
    async def claim(self):
        async with self._admission_lock:
            if self._admitted >= self.limit:
                raise QueueFullError("inference queue is full")
            self._admitted += 1
        try:
            async with self._execution_lock:
                yield
        finally:
            async with self._admission_lock:
                self._admitted -= 1

    async def run(self, work):
        async with self.claim():
            # Only the active request occupies AnyIO's framework-managed worker.
            return await anyio.to_thread.run_sync(work, abandon_on_cancel=False)


def _error(code: ErrorCode, detail: str, status_code: int = 422) -> HTTPException:
    return HTTPException(status_code=status_code, detail=ErrorResult(code=code, detail=detail).model_dump())


async def _read(file: UploadFile, settings: Settings) -> UploadedDocument:
    data = await file.read()
    await file.close()
    if not data:
        raise _error(ErrorCode.INVALID_UPLOAD, "Upload must not be empty")
    if len(data) > settings.batch.max_file_bytes:
        raise _error(ErrorCode.UPLOAD_TOO_LARGE, "Upload exceeds BATCH_MAX_FILE_BYTES", 413)
    return document_from_bytes(data, file.filename, file.content_type)


def _image(document: UploadedDocument) -> UploadedDocument:
    try:
        image_from_document(document)
    except HTTPException as exc:
        raise _error(ErrorCode.INVALID_UPLOAD, str(exc.detail)) from exc
    return document


def _safe_entries(archive: UploadedDocument, settings: Settings) -> list[UploadedDocument]:
    if archive.extension != "zip":
        raise _error(ErrorCode.INVALID_ARCHIVE, "archive must be a ZIP file")
    try:
        with zipfile.ZipFile(BytesIO(archive.data)) as zipped:
            entries = []
            total = 0
            for info in zipped.infolist():
                path = PurePosixPath(info.filename)
                if info.is_dir() or "__MACOSX" in path.parts or path.name in {".DS_Store", "Thumbs.db"}:
                    continue
                if path.is_absolute() or ".." in path.parts or not path.name:
                    raise _error(ErrorCode.INVALID_ARCHIVE, "ZIP contains an unsafe path")
                if info.flag_bits & 1:
                    raise _error(ErrorCode.INVALID_ARCHIVE, "Encrypted ZIP entries are not supported")
                if info.file_size > settings.batch.max_file_bytes:
                    raise _error(ErrorCode.UPLOAD_TOO_LARGE, "ZIP entry exceeds BATCH_MAX_FILE_BYTES", 413)
                total += info.file_size
                if total > settings.batch.max_archive_uncompressed_bytes:
                    raise _error(ErrorCode.UPLOAD_TOO_LARGE, "ZIP exceeds BATCH_MAX_ARCHIVE_UNCOMPRESSED_BYTES", 413)
                entries.append(
                    document_from_bytes(
                        zipped.read(info), path.name, source_filename=archive.filename, archive_path=info.filename
                    )
                )
                if len(entries) > settings.batch.max_files * 2:
                    raise _error(ErrorCode.TOO_MANY_ITEMS, "ZIP contains too many files", 413)
    except zipfile.BadZipFile as exc:
        raise _error(ErrorCode.INVALID_ARCHIVE, "archive is not a valid ZIP file") from exc
    return entries


def _id_archive(archive: UploadedDocument, settings: Settings) -> list[LogicalInput]:
    pairs: dict[str, dict[str, UploadedDocument]] = {}
    for entry in _safe_entries(archive, settings):
        path = PurePosixPath(entry.archive_path or entry.filename or "")
        if not path.parent.parts:
            raise _error(ErrorCode.MISSING_DOCUMENT_SIDE, "each ID-card image must be in a card directory")
        side = path.stem.lower()
        if side not in {"front", "back"}:
            raise _error(ErrorCode.INVALID_ARCHIVE, "ID-card filenames must be front.* or back.*")
        if entry.extension not in {"jpg", "jpeg", "png"}:
            raise _error(ErrorCode.INVALID_ARCHIVE, "ID-card sides must be supported image files")
        card = str(path.parent)
        if side in pairs.setdefault(card, {}):
            raise _error(ErrorCode.INVALID_ARCHIVE, f"ID-card directory {card!r} has duplicate {side} images")
        pairs[card][side] = entry
    results = []
    for card, sides in pairs.items():
        if set(sides) != {"front", "back"}:
            raise _error(ErrorCode.MISSING_DOCUMENT_SIDE, f"ID-card directory {card!r} requires front and back")
        results.append(LogicalInput(DocumentInput(document_type=DocumentType.ID_CARD, front=f"{card}/front", back=f"{card}/back"), (sides["front"], sides["back"])))
    return results


def _image_inputs(document_type: DocumentType, documents: list[UploadedDocument]) -> list[LogicalInput]:
    return [LogicalInput(DocumentInput(document_type=document_type, image=file.filename or "image"), (file,)) for file in documents]


def _single(response: OcrBatchResponse) -> OcrResponse:
    item = response.items[0]
    if not item.success:
        raise _error(item.error.code, item.error.detail)
    return OcrResponse(result=item.result)


def _run_batch(inputs: list[LogicalInput], models: Models, settings: Settings) -> OcrBatchResponse:
    if not inputs:
        raise _error(ErrorCode.INVALID_UPLOAD, "At least one image is required")
    if len(inputs) > settings.batch.max_files:
        raise _error(ErrorCode.TOO_MANY_ITEMS, "Batch exceeds BATCH_MAX_FILES", 413)
    started = time.perf_counter()
    artifacts = create_batch_artifact_run(settings.artifacts, "v1")
    artifacts.save_json(
        "00_request.json",
        {"inputs": [logical.input.model_dump(mode="json") for logical in inputs]},
    )
    planning_started = time.perf_counter()
    plan = build_batch_plan(inputs, settings, artifacts)
    planning_seconds = time.perf_counter() - planning_started
    try:
        outcomes, diagnostics = models.profile_batch_runner().run(plan.jobs) if plan.jobs else ([], {})
    except QueueFullError as exc:
        raise _error(ErrorCode.QUEUE_FULL, str(exc), 503) from exc
    except ResourceExhaustedError as exc:
        raise _error(ErrorCode.RESOURCE_EXHAUSTED, str(exc), 503) from exc
    if os.getenv("VOIGHT_BENCHMARK_TRACE_DIR"):
        diagnostics["benchmark_input_preparation_seconds"] = planning_seconds
    response = assemble_batch_response(inputs, plan, outcomes, diagnostics, started)
    artifacts.save_json("01_pipeline_diagnostics.json", response.diagnostics)
    artifacts.save_json("02_response.json", response.model_dump(mode="json"))
    return response


def create_v1_router(settings: Settings, models: Models) -> APIRouter:
    router = APIRouter(prefix="/v1")
    gate = AsyncInferenceGate(settings.runtime.queue_limit)

    async def run(inputs: list[LogicalInput]) -> OcrBatchResponse:
        try:
            return await gate.run(lambda: _run_batch(inputs, models, settings))
        except QueueFullError as exc:
            raise _error(ErrorCode.QUEUE_FULL, str(exc), 503) from exc

    @router.post("/ocr/passport", response_model=OcrResponse, tags=["v1 OCR"])
    async def passport(image: UploadFile = File(...)) -> OcrResponse:
        return _single(await run(_image_inputs(DocumentType.PASSPORT, [await _read(image, settings)])))

    @router.post("/ocr/id-card", response_model=OcrResponse, tags=["v1 OCR"])
    async def id_card(front: UploadFile = File(...), back: UploadFile = File(...)) -> OcrResponse:
        files = (await _read(front, settings), await _read(back, settings))
        response = await run([LogicalInput(DocumentInput(document_type=DocumentType.ID_CARD, front=files[0].filename or "front", back=files[1].filename or "back"), files)])
        return _single(response)

    @router.post("/ocr/driving-license", response_model=OcrResponse, tags=["v1 OCR"])
    async def driving_license(image: UploadFile = File(...)) -> OcrResponse:
        return _single(await run(_image_inputs(DocumentType.DRIVING_LICENSE, [await _read(image, settings)])))

    @router.post("/ocr/passport/batch", response_model=OcrBatchResponse, tags=["v1 OCR"])
    async def passport_batch(images: BatchUploads = [], archive: OptionalUpload = None) -> OcrBatchResponse:
        files = [await _read(file, settings) for file in images]
        if archive is not None:
            files.extend(_safe_entries(await _read(archive, settings), settings))
        return await run(_image_inputs(DocumentType.PASSPORT, files))

    @router.post("/ocr/id-card/batch", response_model=OcrBatchResponse, tags=["v1 OCR"])
    async def id_card_batch(archive: UploadFile = File(...)) -> OcrBatchResponse:
        return await run(_id_archive(await _read(archive, settings), settings))

    @router.post("/ocr/driving-license/batch", response_model=OcrBatchResponse, tags=["v1 OCR"])
    async def driving_license_batch(images: BatchUploads = [], archive: OptionalUpload = None) -> OcrBatchResponse:
        files = [await _read(file, settings) for file in images]
        if archive is not None:
            files.extend(_safe_entries(await _read(archive, settings), settings))
        return await run(_image_inputs(DocumentType.DRIVING_LICENSE, files))

    @router.get("/health/live", tags=["Health"])
    async def live() -> dict[str, str]:
        return {"status": "live"}

    @router.get("/health/ready", tags=["Health"])
    async def ready() -> dict:
        try:
            settings.validate_startup()
            models.readiness()
        except (RuntimeError, ValueError) as exc:
            raise _error(ErrorCode.PROFILE_UNAVAILABLE, str(exc), 503) from exc
        return {"status": "ready", "models": models.configuration(), "runtime": models.readiness()}

    return router
