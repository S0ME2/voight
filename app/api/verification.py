"""Separate whole-document OCR and field verification transport."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time

from fastapi import APIRouter, Body, File, HTTPException, UploadFile

from app.api.schemas import (
    VerificationCheckRequest,
    VerificationFieldResult,
    VerificationIdCardCheckRequest,
    VerificationIdCardOcrResponse,
    VerificationOcrBatchItem,
    VerificationOcrBatchResponse,
    VerificationOcrLine,
    VerificationOcrResponse,
    VerificationResponse,
    VerificationSummary,
)
from app.api.upload_types import BatchUploads, OptionalUpload
from app.api.v1 import AsyncInferenceGate, _error, _read, _safe_entries
from app.artifacts import create_artifact_run, create_batch_artifact_run, create_child_artifact_run, save_input
from app.config import Settings
from app.contracts import ErrorCode, ErrorResult
from app.inference import QueueFullError, ResourceExhaustedError
from app.inference.batch import OcrSample
from app.models import Models
from app.uploads import UploadedDocument, image_from_document
from app.verification import VerificationLine, verify_fields


@dataclass(frozen=True)
class _LogicalDocument:
    files: tuple[UploadedDocument, ...]


def _write_benchmark_trace(operation: str, trace: dict) -> None:
    directory = os.getenv("VOIGHT_BENCHMARK_TRACE_DIR")
    if not directory:
        return
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    path = target / f"{operation}-{os.getpid()}-{time.time_ns()}.json"
    path.write_text(json.dumps(trace, sort_keys=True), encoding="utf-8")


def _id_documents(entries: list[UploadedDocument]) -> list[_LogicalDocument]:
    pairs: dict[str, dict[str, UploadedDocument]] = {}
    for entry in entries:
        path = (entry.archive_path or entry.filename or "").replace("\\", "/")
        parent, _, name = path.rpartition("/")
        side = name.rsplit(".", 1)[0].lower()
        if not parent:
            raise _error(ErrorCode.MISSING_DOCUMENT_SIDE, "each ID-card image must be in a card directory")
        if side not in {"front", "back"}:
            raise _error(ErrorCode.INVALID_ARCHIVE, "ID-card filenames must be front.* or back.*")
        card = pairs.setdefault(parent, {})
        if side in card:
            raise _error(ErrorCode.INVALID_ARCHIVE, f"ID-card directory {parent!r} has duplicate {side} images")
        card[side] = entry
    documents = []
    for card, sides in pairs.items():
        if set(sides) != {"front", "back"}:
            raise _error(ErrorCode.MISSING_DOCUMENT_SIDE, f"ID-card directory {card!r} requires front and back")
        documents.append(_LogicalDocument((sides["front"], sides["back"])))
    return documents


def _line_result(tokens: list[dict], side: str) -> list[VerificationOcrLine]:
    return [
        VerificationOcrLine(
            line_id=str(token.get("index", index)),
            text=token["text"],
            confidence=token["score"],
            bbox=(token["x1"], token["y1"], token["x2"], token["y2"]) if all(key in token for key in ("x1", "y1", "x2", "y2")) else None,
            reading_order=index,
            side=side,
        )
        for index, token in enumerate(tokens)
        if token.get("text", "").strip()
    ]


def _run_ocr(documents: list[_LogicalDocument], models: Models, settings: Settings):
    started = time.perf_counter()
    if not documents:
        raise _error(ErrorCode.INVALID_UPLOAD, "At least one image is required")
    if len(documents) > settings.batch.max_files:
        raise _error(ErrorCode.TOO_MANY_ITEMS, "Batch exceeds BATCH_MAX_FILES", 413)
    samples = []
    artifacts = create_batch_artifact_run(settings.artifacts, "verification")
    sample_artifacts = {}
    decoded: list[tuple[object, ...] | None] = []
    errors: dict[int, ErrorResult] = {}
    decode_started = time.perf_counter()
    for index, document in enumerate(documents):
        try:
            images = tuple(image_from_document(file) for file in document.files)
        except HTTPException:
            errors[index] = ErrorResult(code=ErrorCode.INVALID_UPLOAD, detail="Upload must be a decodable image")
            decoded.append(None)
            continue
        decoded.append(images)
        for side_index, image in enumerate(images):
            side = ("front", "back")[side_index] if len(images) == 2 else "image"
            sample_id = f"verification:{index}:{side}"
            artifact = create_child_artifact_run(artifacts, len(sample_artifacts), image.filename)
            save_input(
                artifact,
                image.data,
                image.filename,
                image.content_type,
                image.extension,
                image.image,
                source_filename=image.source_filename,
                archive_path=image.archive_path,
            )
            sample_artifacts[sample_id] = artifact
            samples.append((index, side, image.image, artifact))

    artifacts.save_json(
        "00_request.json",
        {
            "documents": [
                {
                    "index": index,
                    "files": [
                        {
                            "side": ("front", "back")[side_index] if len(document.files) == 2 else "image",
                            "filename": file.filename,
                            "content_type": file.content_type,
                            "extension": file.extension,
                            "source_filename": file.source_filename,
                            "archive_path": file.archive_path,
                        }
                        for side_index, file in enumerate(document.files)
                    ],
                }
                for index, document in enumerate(documents)
            ],
        },
    )

    decode_seconds = time.perf_counter() - decode_started
    inference_started = time.perf_counter()
    ocr_result = models.verification_ocr().run(
        [OcrSample(f"verification:{index}:{side}", image, artifacts=artifact) for index, side, image, artifact in samples]
    ) if samples else None
    if ocr_result is not None:
        artifacts.save_json("01_ocr_diagnostics.json", ocr_result.diagnostics)
        for sample_id, artifact in sample_artifacts.items():
            artifact.save_json("01_ocr_tokens.json", ocr_result.tokens.get(sample_id, []))
            if error := ocr_result.errors.get(sample_id):
                artifact.save_json("02_ocr_error.json", {"error": str(error)})
    if errors:
        artifacts.save_json(
            "02_input_errors.json",
            {str(index): error.model_dump(mode="json") for index, error in errors.items()},
        )
    inference_seconds = time.perf_counter() - inference_started
    assembly_started = time.perf_counter()
    results = []
    for index, images in enumerate(decoded):
        if index in errors:
            results.append(errors[index])
            continue
        if images is None:
            results.append(ErrorResult(code=ErrorCode.INVALID_UPLOAD, detail="Upload must be a decodable image"))
            continue
        values = []
        for side_index in range(len(images)):
            side = ("front", "back")[side_index] if len(images) == 2 else "image"
            sample_id = f"verification:{index}:{side}"
            if sample_id in ocr_result.errors:
                values.append(ErrorResult(code=ErrorCode.PROCESSING_FAILED, detail="OCR failed for this image"))
                break
            values.append(_line_result(ocr_result.tokens.get(sample_id, []), side))
        if values and isinstance(values[0], ErrorResult):
            results.append(values[0])
        elif len(values) == 2:
            results.append(VerificationIdCardOcrResponse(front=values[0], back=values[1]))
        else:
            results.append(VerificationOcrResponse(lines=values[0]))
    artifacts.save_json(
        "03_ocr_results.json",
        [
            value.model_dump(mode="json") if hasattr(value, "model_dump") else value
            for value in results
        ],
    )
    _write_benchmark_trace(
        "ocr",
        {
            "operation": "ocr",
            "image_count": len(samples),
            "document_count": len(documents),
            "image_decode_seconds": decode_seconds,
            "inference_and_assembly_seconds": time.perf_counter() - inference_started,
            "model_inference_seconds": inference_seconds,
            "response_assembly_seconds": time.perf_counter() - assembly_started,
            "total_server_seconds": time.perf_counter() - started,
            "diagnostics": ocr_result.diagnostics if ocr_result is not None else {},
        },
    )
    return results


def _single(value):
    if isinstance(value, ErrorResult):
        raise _error(value.code, value.detail)
    return value


def _check(request, settings: Settings, *, id_card: bool = False, operation: str = "check") -> VerificationResponse:
    started = time.perf_counter()
    artifacts = create_artifact_run(settings.artifacts, "verification_check", operation)
    artifacts.save_json("00_request.json", request.model_dump(mode="json"))
    parse_to_lines_started = time.perf_counter()
    lines = []
    if id_card:
        for side in ("front", "back"):
            lines.extend(
                VerificationLine(
                    line.text,
                    line.confidence,
                    side,
                    line.line_id,
                    line.bbox,
                    line.polygon,
                    line.reading_order,
                    side,
                )
                for line in getattr(request.ocr, side)
            )
    else:
        lines = [
            VerificationLine(line.text, line.confidence, "visible_ocr", line.line_id, line.bbox, line.polygon, line.reading_order, line.side)
            for line in request.ocr.lines
        ]
    parse_to_lines_seconds = time.perf_counter() - parse_to_lines_started
    artifacts.save_json(
        "01_normalized_ocr_lines.json",
        [
            {
                "text": line.text,
                "confidence": line.confidence,
                "source": line.source,
                "line_id": line.line_id,
                "bbox": line.bbox,
                "polygon": line.polygon,
                "reading_order": line.reading_order,
                "side": line.side,
            }
            for line in lines
        ],
    )
    timings: dict[str, float] = {}
    match_started = time.perf_counter()
    value = verify_fields(lines, request.fields, document_type="id_card" if id_card else "passport" if operation == "passport" else None, instrumentation=timings)
    match_seconds = time.perf_counter() - match_started
    assembly_started = time.perf_counter()
    response = VerificationResponse(
        fields={name: VerificationFieldResult(**field) for name, field in value["fields"].items()},
        summary=VerificationSummary(**value["summary"]),
    )
    artifacts.save_json("02_matcher_result.json", value)
    artifacts.save_json("03_response.json", response.model_dump(mode="json"))
    artifacts.save_json(
        "04_timing.json",
        {
            "normalization_and_line_conversion_seconds": parse_to_lines_seconds,
            "matching_seconds": match_seconds,
            "total_server_seconds": time.perf_counter() - started,
            "instrumentation": timings,
        },
    )
    _write_benchmark_trace(
        "check",
        {
            "operation": operation,
            "line_count": len(lines),
            "field_count": len(request.fields),
            "payload_parsing_seconds": None,
            "normalization_and_line_conversion_seconds": parse_to_lines_seconds,
            "candidate_generation_seconds": timings.get("candidate_generation_seconds", 0.0),
            "token_span_construction_seconds": timings.get("token_span_construction_seconds", 0.0),
            "geometry_assembly_seconds": timings.get("geometry_assembly_seconds", 0.0),
            "mrz_validation_seconds": timings.get("mrz_validation_seconds", 0.0),
            "normalization_seconds": timings.get("normalization_seconds", 0.0),
            "similarity_scoring_seconds": timings.get("similarity_scoring_seconds", 0.0),
            "assignment_seconds": timings.get("assignment_seconds", 0.0),
            "status_classification_seconds": timings.get("status_classification_seconds", 0.0),
            "strict_field_validation_seconds": timings.get("strict_field_validation_seconds", 0.0),
            "matching_seconds": match_seconds,
            "response_assembly_seconds": time.perf_counter() - assembly_started,
            "candidate_count": timings.get("candidate_count", 0),
            "score_comparison_count": timings.get("score_comparison_count", 0),
            "option_counts": timings.get("option_counts", []),
            "candidate_evidence": timings.get("candidate_evidence", []),
            "assignment_states": timings.get("assignment_states", 0),
            "assignment_transitions": timings.get("assignment_transitions", 0),
            "total_server_seconds": time.perf_counter() - started,
        },
    )
    return response


def create_verification_router(settings: Settings, models: Models) -> APIRouter:
    router = APIRouter(prefix="/verification")
    gate = AsyncInferenceGate(settings.runtime.queue_limit)

    async def read_image(file: UploadFile) -> UploadedDocument:
        return await _read(file, settings)

    async def run(documents: list[_LogicalDocument]):
        try:
            return await gate.run(lambda: _run_ocr(documents, models, settings))
        except QueueFullError as exc:
            raise _error(ErrorCode.QUEUE_FULL, str(exc), 503) from exc
        except ResourceExhaustedError as exc:
            raise _error(ErrorCode.RESOURCE_EXHAUSTED, str(exc), 503) from exc

    def image_route(path: str, *, id_card: bool = False):
        if id_card:
            async def endpoint(front: UploadFile = File(...), back: UploadFile = File(...)):
                values = await run([_LogicalDocument((await read_image(front), await read_image(back)))])
                return _single(values[0])
        else:
            async def endpoint(image: UploadFile = File(...)):
                values = await run([_LogicalDocument((await read_image(image),))])
                return _single(values[0])
        endpoint.__name__ = path.replace("/", "_").strip("_")
        router.add_api_route(
            path,
            endpoint,
            methods=["POST"],
            response_model=VerificationIdCardOcrResponse if id_card else VerificationOcrResponse,
            tags=["Verification OCR"],
        )

    image_route("/passport/ocr")
    image_route("/id-card/ocr", id_card=True)
    image_route("/driving-licence/ocr")

    async def ordinary_batch(images: BatchUploads = [], archive: OptionalUpload = None):
        files = [await read_image(file) for file in images]
        if archive is not None:
            archive_document = await read_image(archive)
            files.extend(_safe_entries(archive_document, settings))
        return await make_batch_response([_LogicalDocument((file,)) for file in files])

    async def id_batch(archive: UploadFile = File(...)):
        return await make_batch_response(_id_documents(_safe_entries(await read_image(archive), settings)))

    async def ordinary_batch_route(images: BatchUploads = [], archive: OptionalUpload = None):
        return await ordinary_batch(images, archive)

    async def driving_batch_route(images: BatchUploads = [], archive: OptionalUpload = None):
        return await ordinary_batch(images, archive)

    router.add_api_route("/passport/ocr/batch", ordinary_batch_route, methods=["POST"], response_model=VerificationOcrBatchResponse, operation_id="passport_verification_ocr_batch", tags=["Verification OCR"])
    router.add_api_route("/id-card/ocr/batch", id_batch, methods=["POST"], response_model=VerificationOcrBatchResponse, operation_id="id_card_verification_ocr_batch", tags=["Verification OCR"])
    router.add_api_route("/driving-licence/ocr/batch", driving_batch_route, methods=["POST"], response_model=VerificationOcrBatchResponse, operation_id="driving_licence_verification_ocr_batch", tags=["Verification OCR"])

    async def make_batch_response(documents):
        values = await run(documents)
        return VerificationOcrBatchResponse(
            total=len(values),
            succeeded=sum(not isinstance(value, ErrorResult) for value in values),
            failed=sum(isinstance(value, ErrorResult) for value in values),
            items=[
                VerificationOcrBatchItem(index=index, success=not isinstance(value, ErrorResult), result=None if isinstance(value, ErrorResult) else value, error=value if isinstance(value, ErrorResult) else None)
                for index, value in enumerate(values)
            ],
        )

    @router.post("/passport/check", response_model=VerificationResponse, tags=["Verification checks"])
    async def passport_check(request: VerificationCheckRequest = Body(...)) -> VerificationResponse:
        return _check(request, settings, operation="passport")

    @router.post("/driving-licence/check", response_model=VerificationResponse, tags=["Verification checks"])
    async def driving_check(request: VerificationCheckRequest = Body(...)) -> VerificationResponse:
        return _check(request, settings, operation="driving_license")

    @router.post("/id-card/check", response_model=VerificationResponse, tags=["Verification checks"])
    async def id_check(request: VerificationIdCardCheckRequest = Body(...)) -> VerificationResponse:
        return _check(request, settings, id_card=True, operation="id_card")

    return router
