"""Frozen v1 HTTP transport, kept separate from migration-era routes."""

from __future__ import annotations

import time
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.schemas import OcrBatchResponse, OcrResponse
from app.artifacts import create_batch_artifact_run, create_child_artifact_run, save_input
from app.api.upload_types import BatchUploads, OptionalUpload
from app.config import Settings
from app.contracts import (
    BatchItemResult,
    DocumentInput,
    DocumentResult,
    DocumentType,
    ErrorCode,
    ErrorResult,
    FieldResult,
    TimingResult,
)
from app.documents.driving_license_fields import parse_fields as parse_license_fields
from app.documents.driving_license_fields import validation_warnings
from app.documents.identity import (
    DocumentPipelineError,
    _document_confidence,
    _field_results,
    _parse_visible,
    _reconcile,
    _required_validation,
    _required_warnings,
)
from app.documents.mrz import ID_CARD, PASSPORT, parse as parse_mrz
from app.documents.passport_localization import detect_passport_page_padded
from app.documents.profiles import load_document_profile
from app.inference import ProfileBatchItem, QueueFullError
from app.models import Models
from app.pipeline import RegionProfile, load_region_profile
from app.uploads import UploadedDocument, document_from_bytes, image_from_document
from app.workflows import run_document_mrz


@dataclass(frozen=True)
class LogicalInput:
    input: DocumentInput
    files: tuple[UploadedDocument, ...]


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


def _item_error(index: int, source: DocumentInput, error: ErrorResult) -> BatchItemResult:
    return BatchItemResult(index=index, input=source, success=False, error=error)


def _single(response: OcrBatchResponse) -> OcrResponse:
    item = response.items[0]
    if not item.success:
        raise _error(item.error.code, item.error.detail)
    return OcrResponse(result=item.result)


def _driving_field_results(values: dict, report: dict) -> dict[str, FieldResult]:
    source_names = {
        "surname": ("surname",),
        "given_names": ("given_names", "name"),
        "birth_place": ("birth_place_and_date", "place_of_birth"),
        "birth_date": ("date_of_birth",),
        "issue_date": ("issue_date", "date_of_issue"),
        "expiry_date": ("expiry_date", "date_of_expiry"),
        "issued_place": ("issued_place", "place_of_issue"),
        "personal_id": ("personal_id", "id_number"),
        "license_number": ("license_number", "id_number_2"),
        "address": ("address", "place_of_living"),
        "categories": ("categories", "types"),
        "serial_number": ("serial_number",),
    }

    def evidence(name: str, kind: str):
        for source in source_names.get(name, (name,)):
            if source in report[kind]:
                return report[kind][source]
        return [] if kind == "field_raw_text" else None

    return {
        name: FieldResult(value=value, raw_text=evidence(name, "field_raw_text"), confidence=evidence(name, "field_confidences"), bounding_box=evidence(name, "field_bounding_boxes"), region="image")
        for name, value in values.items()
    }


def _mrz_result(upload, document_type: DocumentType, models: Models, settings: Settings, artifacts):
    profile = PASSPORT if document_type == DocumentType.PASSPORT else ID_CARD
    try:
        output = run_document_mrz(upload, "v1_mrz", profile, models, settings, artifacts=artifacts, save_input_artifacts=False)
    except Exception:
        return parse_mrz("", document_type.value)
    return parse_mrz(str(output.result), document_type.value)


def _run_batch(inputs: list[LogicalInput], models: Models, settings: Settings) -> OcrBatchResponse:
    if not inputs:
        raise _error(ErrorCode.INVALID_UPLOAD, "At least one image is required")
    if len(inputs) > settings.batch.max_files:
        raise _error(ErrorCode.TOO_MANY_ITEMS, "Batch exceeds BATCH_MAX_FILES", 413)
    started = time.perf_counter()
    artifacts = create_batch_artifact_run(settings.artifacts, "v1")
    jobs: list[ProfileBatchItem] = []
    owners: list[tuple[int, str, dict, object]] = []
    mrz_inputs = {}
    item_errors: dict[int, ErrorResult] = {}
    aligner = None
    mrz_detector = None
    for index, logical in enumerate(inputs):
        try:
            decoded_files = tuple(image_from_document(file) for file in logical.files)
        except HTTPException:
            item_errors[index] = ErrorResult(code=ErrorCode.INVALID_UPLOAD, detail="Upload must be a decodable image")
            continue
        if logical.input.document_type == DocumentType.DRIVING_LICENSE:
            profile = load_region_profile(settings.driving_license.data_crop, settings.driving_license.field_rois)
            document_profile = None
            regions = [("image", decoded_files[0], profile, settings.driving_license.canonical_width, settings.driving_license.canonical_height, parse_license_fields, validation_warnings)]
        else:
            profile_path = settings.profiles.passport if logical.input.document_type == DocumentType.PASSPORT else settings.profiles.id_card
            document_profile = load_document_profile(profile_path)
            region_names = ("data_page",) if logical.input.document_type == DocumentType.PASSPORT else ("front", "back")
            regions = [(region, file, RegionProfile(document_profile["regions"][region]["data_crop"], document_profile["regions"][region]["field_rois"]), int(document_profile["canonical_size"]["width"]), int(document_profile["canonical_size"]["height"]), _parse_visible, _required_warnings(document_profile, region)) for region, file in zip(region_names, decoded_files)]
        for region, file, profile, width, height, parser, validator in regions:
            image = file.image
            writer = create_child_artifact_run(artifacts, len(jobs), file.filename)
            save_input(writer, file.data, file.filename, file.content_type, file.extension, image, source_filename=file.source_filename, archive_path=file.archive_path)
            item_id = f"{index}:{region}"
            if logical.input.document_type == DocumentType.PASSPORT:
                mrz_detector = mrz_detector or models.mrz_scanner()
                detector = lambda padded, document_profile=document_profile, model=mrz_detector: detect_passport_page_padded(padded, document_profile, model, settings.driving_license.aligner_padding)
            else:
                aligner = aligner or models.document_aligner()
                detector = lambda padded, model=aligner: model(img=padded, do_center_crop=False)
            jobs.append(ProfileBatchItem(item_id, image, profile, detector, parser, validator, writer, width, height, settings.driving_license.aligner_padding, settings.driving_license.min_overlap_ratio))
            owners.append((index, region, document_profile, logical.input.document_type))
            mrz_inputs[index, region] = (file, writer)
    try:
        outcomes, _ = models.profile_batch_runner().run(jobs) if jobs else ([], {})
    except QueueFullError as exc:
        raise _error(ErrorCode.QUEUE_FULL, str(exc), 503) from exc
    grouped: dict[int, dict[str, tuple[object, dict | None]]] = {}
    for owner, outcome in zip(owners, outcomes):
        index, region, profile, kind = owner
        grouped.setdefault(index, {})[region] = (outcome, profile)
    items = []
    for index, logical in enumerate(inputs):
        if index in item_errors:
            items.append(_item_error(index, logical.input, item_errors[index]))
            continue
        regions = grouped.get(index, {})
        failed = next((outcome.error for outcome, _ in regions.values() if outcome.error), None)
        if failed:
            code = failed.error.code if isinstance(failed, DocumentPipelineError) else ErrorCode.INVALID_DOCUMENT
            detail = failed.error.detail if isinstance(failed, DocumentPipelineError) else "document could not be processed"
            items.append(_item_error(index, logical.input, ErrorResult(code=code, detail=detail)))
            continue
        if logical.input.document_type == DocumentType.DRIVING_LICENSE:
            values, report = regions["image"][0].result
            fields = _driving_field_results(values, report)
            result = DocumentResult(document_type=DocumentType.DRIVING_LICENSE, layout="driving_license", fields=fields, warnings=report["validation_warnings"], timings=TimingResult(total_seconds=report["timings"]["total_seconds"], stages=report["timings"]))
        else:
            profile = next(profile for _, profile in regions.values() if profile is not None)
            fields = {}
            warnings = []
            for region, (outcome, _) in regions.items():
                values, report = outcome.result
                fields.update(_field_results(profile, region, values, report))
                warnings.extend(report["validation_warnings"])
            mrz_region = "data_page" if logical.input.document_type == DocumentType.PASSPORT else "back"
            upload, writer = mrz_inputs[index, mrz_region]
            mrz = _mrz_result(upload, logical.input.document_type, models, settings, writer)
            mapping = {"surname": "surname", "name": "given_names", "date_of_birth": "date_of_birth", "sex": "sex", "date_of_expiry": "date_of_expiry", "passport_number" if logical.input.document_type == DocumentType.PASSPORT else "card_number": "document_number"}
            if logical.input.document_type == DocumentType.ID_CARD:
                mapping.update({"pinfl": "pinfl", "citizenship": "nationality"})
            validations = [_required_validation(profile, fields)] + _reconcile(fields, mrz.fields, mapping)
            if not mrz.raw_lines:
                warnings.append("MRZ was not found" + (" on ID-card back" if logical.input.document_type == DocumentType.ID_CARD else ""))
            elif any(validation.status.value == "failed" for validation in mrz.validations):
                warnings.append("MRZ check-digit validation failed")
            result = DocumentResult(document_type=logical.input.document_type, layout=profile["layout"], fields=fields, document_confidence=_document_confidence([outcome.result[1] for outcome, _ in regions.values()]), mrz=mrz, validations=validations, warnings=warnings, timings=TimingResult(total_seconds=max((outcome.result[1]["timings"]["total_seconds"] for outcome, _ in regions.values()), default=0.0)))
        items.append(BatchItemResult(index=index, input=logical.input, success=True, result=result))
    return OcrBatchResponse(total=len(items), succeeded=sum(item.success for item in items), failed=sum(not item.success for item in items), total_seconds=time.perf_counter() - started, items=items)


def create_v1_router(settings: Settings, models: Models) -> APIRouter:
    router = APIRouter(prefix="/v1")

    @router.post("/ocr/passport", response_model=OcrResponse)
    async def passport(image: UploadFile = File(...)) -> OcrResponse:
        return _single(_run_batch(_image_inputs(DocumentType.PASSPORT, [await _read(image, settings)]), models, settings))

    @router.post("/ocr/id-card", response_model=OcrResponse)
    async def id_card(front: UploadFile = File(...), back: UploadFile = File(...)) -> OcrResponse:
        files = (await _read(front, settings), await _read(back, settings))
        response = _run_batch([LogicalInput(DocumentInput(document_type=DocumentType.ID_CARD, front=files[0].filename or "front", back=files[1].filename or "back"), files)], models, settings)
        return _single(response)

    @router.post("/ocr/driving-license", response_model=OcrResponse)
    async def driving_license(image: UploadFile = File(...)) -> OcrResponse:
        return _single(_run_batch(_image_inputs(DocumentType.DRIVING_LICENSE, [await _read(image, settings)]), models, settings))

    @router.post("/ocr/passport/batch", response_model=OcrBatchResponse)
    async def passport_batch(images: BatchUploads = [], archive: OptionalUpload = None) -> OcrBatchResponse:
        files = [await _read(file, settings) for file in images]
        if archive is not None:
            files.extend(_safe_entries(await _read(archive, settings), settings))
        return _run_batch(_image_inputs(DocumentType.PASSPORT, files), models, settings)

    @router.post("/ocr/id-card/batch", response_model=OcrBatchResponse)
    async def id_card_batch(archive: UploadFile = File(...)) -> OcrBatchResponse:
        return _run_batch(_id_archive(await _read(archive, settings), settings), models, settings)

    @router.post("/ocr/driving-license/batch", response_model=OcrBatchResponse)
    async def driving_license_batch(images: BatchUploads = [], archive: OptionalUpload = None) -> OcrBatchResponse:
        files = [await _read(file, settings) for file in images]
        if archive is not None:
            files.extend(_safe_entries(await _read(archive, settings), settings))
        return _run_batch(_image_inputs(DocumentType.DRIVING_LICENSE, files), models, settings)

    @router.get("/health/live")
    async def live() -> dict[str, str]:
        return {"status": "live"}

    @router.get("/health/ready")
    async def ready() -> dict[str, str]:
        try:
            settings.validate_startup()
        except ValueError as exc:
            raise _error(ErrorCode.PROFILE_UNAVAILABLE, str(exc), 503) from exc
        return {"status": "ready"}

    return router
