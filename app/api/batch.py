import logging
import time
import zipfile
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any

from fastapi import HTTPException, UploadFile, status

from app.api.schemas import BatchItemError, BatchItemResponse, BatchResponse
from app.artifacts import (
    ArtifactWriter,
    create_batch_artifact_run,
    create_child_artifact_run,
    save_input,
)
from app.config import Settings
from app.uploads import UploadedDocument, UploadedImage, document_from_bytes, image_from_document
from app.workflows import WorkflowResult

logger = logging.getLogger(__name__)

BatchProcessor = Callable[[UploadedImage, ArtifactWriter], WorkflowResult]


@dataclass(frozen=True)
class BatchInput:
    upload: UploadedDocument
    preparation_seconds: float


async def process_image_batch(
    files: Sequence[UploadFile],
    processor: BatchProcessor,
    *,
    operation: str,
    settings: Settings,
) -> BatchResponse:
    """Process direct images and ZIP entries sequentially with nested artifacts.

    Sequential execution is intentional: the shared OCR/detection instances are
    expensive and are not assumed to be safe for concurrent inference.
    """

    _validate_request(files)
    started_at = datetime.now(timezone.utc)
    started_total = time.perf_counter()
    batch_artifacts = create_batch_artifact_run(settings.artifacts, operation)
    batch_items: list[BatchItemResponse] = []
    item_timing_records: list[dict[str, Any]] = []
    input_preparation_seconds = 0.0

    try:
        preparation_started = time.perf_counter()
        inputs = await _expand_inputs(files, settings)
        input_preparation_seconds = time.perf_counter() - preparation_started
        _validate_expanded_batch(inputs, settings.batch.max_files)

        batch_artifacts.save_json(
            "batch_metadata.json",
            {
                "batch_run_id": batch_artifacts.run_id,
                "operation": operation,
                "started_at_utc": started_at.isoformat(),
                "preload_enabled": settings.preload,
                "top_level_upload_count": len(files),
                "expanded_image_count": len(inputs),
                "top_level_filenames": [file.filename for file in files],
            },
        )

        succeeded = 0
        for index, batch_input in enumerate(inputs):
            item = _process_one(
                batch_input,
                index,
                processor,
                operation,
                batch_artifacts,
            )
            batch_items.append(item[0])
            item_timing_records.append(item[1])
            if item[0].success:
                succeeded += 1

        total_seconds = time.perf_counter() - started_total
        response = BatchResponse(
            total=len(batch_items),
            succeeded=succeeded,
            failed=len(batch_items) - succeeded,
            total_seconds=total_seconds,
            batch_run_id=batch_artifacts.run_id if batch_artifacts.enabled else None,
            items=batch_items,
        )
        _save_batch_summary(
            batch_artifacts,
            response,
            item_timing_records,
            started_at,
            input_preparation_seconds,
            settings.preload,
        )
        return response
    except Exception as exc:
        total_seconds = time.perf_counter() - started_total
        batch_artifacts.save_json(
            "batch_error.json",
            {
                "type": type(exc).__name__,
                "detail": str(exc) if isinstance(exc, HTTPException) else "Batch processing failed",
            },
        )
        batch_artifacts.save_json(
            "timing.json",
            {
                "started_at_utc": started_at.isoformat(),
                "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "total_seconds": total_seconds,
                "input_preparation_seconds": input_preparation_seconds,
                "status": "failed",
                "preload_enabled": settings.preload,
                "model_loading_note": _model_loading_note(settings.preload),
            },
        )
        raise


def _process_one(
    batch_input: BatchInput,
    index: int,
    processor: BatchProcessor,
    operation: str,
    batch_artifacts: ArtifactWriter,
) -> tuple[BatchItemResponse, dict[str, Any]]:
    upload = batch_input.upload
    item_artifacts = create_child_artifact_run(
        batch_artifacts,
        index,
        upload.filename,
    )
    item_started = time.perf_counter()
    decode_seconds = 0.0
    processing_seconds = 0.0
    workflow_timings: dict[str, Any] = {}
    error: BatchItemError | None = None
    result: str | dict[str, Any] | None = None
    success = False
    fatal_error: MemoryError | None = None

    decode_started = time.perf_counter()
    try:
        image_upload = image_from_document(upload)
    except HTTPException as exc:
        decode_seconds = time.perf_counter() - decode_started
        error = BatchItemError(
            status_code=exc.status_code,
            code="invalid_upload",
            detail=str(exc.detail),
        )
        save_input(
            item_artifacts,
            upload.data,
            upload.filename,
            upload.content_type,
            upload.extension,
            source_filename=upload.source_filename,
            archive_path=upload.archive_path,
        )
        item_artifacts.save_json("error.json", error.model_dump())
    else:
        decode_seconds = time.perf_counter() - decode_started
        save_input(
            item_artifacts,
            image_upload.data,
            image_upload.filename,
            image_upload.content_type,
            image_upload.extension,
            image_upload.image,
            source_filename=image_upload.source_filename,
            archive_path=image_upload.archive_path,
        )

        processing_started = time.perf_counter()
        try:
            outcome = processor(image_upload, item_artifacts)
            result = outcome.result
            workflow_timings = outcome.timings
            success = True
        except HTTPException as exc:
            error = BatchItemError(
                status_code=exc.status_code,
                code="processing_rejected",
                detail=str(exc.detail),
            )
            item_artifacts.save_json("error.json", error.model_dump())
        except MemoryError as exc:
            logger.exception(
                "Batch operation %s ran out of memory while processing file %r",
                operation,
                upload.filename,
            )
            fatal_error = exc
            error = BatchItemError(
                status_code=500,
                code="out_of_memory",
                detail="Document processing ran out of memory",
            )
            item_artifacts.save_json("error.json", error.model_dump())
        except Exception:
            logger.exception(
                "Batch operation %s failed while processing file %r",
                operation,
                upload.filename,
            )
            error = BatchItemError(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                code="processing_failed",
                detail="Document processing failed",
            )
            item_artifacts.save_json("error.json", error.model_dump())
        finally:
            processing_seconds = time.perf_counter() - processing_started

    item_seconds = time.perf_counter() - item_started
    total_with_preparation = item_seconds + batch_input.preparation_seconds
    timing = {
        "input_preparation_seconds": batch_input.preparation_seconds,
        "image_decode_seconds": decode_seconds,
        "processing_seconds": processing_seconds,
        "item_handler_seconds": item_seconds,
        "total_including_input_preparation_seconds": total_with_preparation,
        "model_loading_seconds_during_item": _model_loading_seconds(workflow_timings),
        "workflow": workflow_timings,
        "status": "success" if success else "failed",
    }
    item_artifacts.save_json("timing.json", timing)

    if fatal_error is not None:
        raise fatal_error

    response = BatchItemResponse(
        index=index,
        filename=upload.filename,
        source_filename=upload.source_filename,
        archive_path=upload.archive_path,
        success=success,
        result=result,
        error=error,
        total_seconds=total_with_preparation,
    )
    return response, timing


async def _expand_inputs(
    files: Sequence[UploadFile],
    settings: Settings,
) -> list[BatchInput]:
    expanded: list[BatchInput] = []

    for file in files:
        read_started = time.perf_counter()
        try:
            data = await file.read()
        finally:
            await _close_upload(file, "batch_input")
        read_seconds = time.perf_counter() - read_started

        if len(data) > settings.batch.max_file_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Upload {file.filename!r} is {len(data)} bytes; maximum is "
                    f"{settings.batch.max_file_bytes}"
                ),
            )

        document = document_from_bytes(data, file.filename, file.content_type)
        if document.extension == "zip":
            _expand_zip(
                document,
                read_seconds,
                expanded,
                settings,
            )
        else:
            expanded.append(
                BatchInput(
                    upload=document_from_bytes(
                        data,
                        file.filename,
                        file.content_type,
                        source_filename=file.filename,
                    ),
                    preparation_seconds=read_seconds,
                )
            )
        if len(expanded) > settings.batch.max_files:
            _raise_too_many(len(expanded), settings.batch.max_files)

    return expanded


def _expand_zip(
    archive: UploadedDocument,
    archive_read_seconds: float,
    expanded: list[BatchInput],
    settings: Settings,
) -> None:
    try:
        with zipfile.ZipFile(BytesIO(archive.data)) as zipped:
            candidates = [
                info
                for info in zipped.infolist()
                if not info.is_dir() and not _is_ignored_archive_entry(info.filename)
            ]
            total_uncompressed = sum(info.file_size for info in candidates)
            if total_uncompressed > settings.batch.max_archive_uncompressed_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"ZIP {archive.filename!r} expands to {total_uncompressed} bytes; "
                        f"maximum is {settings.batch.max_archive_uncompressed_bytes}"
                    ),
                )

            archive_share = archive_read_seconds / max(1, len(candidates))
            for info in candidates:
                if info.flag_bits & 0x1:
                    raise HTTPException(
                        status_code=422,
                        detail=f"Encrypted ZIP entries are not supported: {info.filename}",
                    )
                if info.file_size > settings.batch.max_file_bytes:
                    raise HTTPException(
                        status_code=413,
                        detail=(
                            f"ZIP entry {info.filename!r} is {info.file_size} bytes; "
                            f"maximum is {settings.batch.max_file_bytes}"
                        ),
                    )
                extraction_started = time.perf_counter()
                entry_data = zipped.read(info)
                extraction_seconds = time.perf_counter() - extraction_started
                expanded.append(
                    BatchInput(
                        upload=document_from_bytes(
                            entry_data,
                            PurePosixPath(info.filename).name,
                            source_filename=archive.filename,
                            archive_path=info.filename,
                        ),
                        preparation_seconds=archive_share + extraction_seconds,
                    )
                )
                if len(expanded) > settings.batch.max_files:
                    _raise_too_many(len(expanded), settings.batch.max_files)
    except zipfile.BadZipFile as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Upload {archive.filename!r} is not a valid ZIP archive",
        ) from exc


def _is_ignored_archive_entry(name: str) -> bool:
    path = PurePosixPath(name)
    return "__MACOSX" in path.parts or path.name in {".DS_Store", "Thumbs.db"}


def _save_batch_summary(
    artifacts: ArtifactWriter,
    response: BatchResponse,
    item_timings: list[dict[str, Any]],
    started_at: datetime,
    input_preparation_seconds: float,
    preload_enabled: bool,
) -> None:
    response_payload = response.model_dump()
    artifacts.save_json("batch_result.json", response_payload)

    previous_model_initialization: dict[str, float] = {}
    for item in item_timings:
        workflow = item.get("workflow", {})
        for key, value in workflow.items():
            suffix = "_recorded_initial_load_seconds"
            if key.endswith(suffix) and isinstance(value, (int, float)):
                previous_model_initialization[key[: -len(suffix)]] = float(value)

    artifacts.save_json(
        "timing.json",
        {
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
            "total_seconds": response.total_seconds,
            "input_preparation_seconds": input_preparation_seconds,
            "sum_item_seconds_including_preparation": sum(
                float(item["total_including_input_preparation_seconds"])
                for item in item_timings
            ),
            "model_loading_seconds_during_batch": sum(
                float(item["model_loading_seconds_during_item"])
                for item in item_timings
            ),
            "recorded_model_initialization_seconds": previous_model_initialization,
            "preload_enabled": preload_enabled,
            "model_loading_note": _model_loading_note(preload_enabled),
            "total_items": response.total,
            "succeeded": response.succeeded,
            "failed": response.failed,
            "status": "completed",
            "timing_scope": "Route-handler wall time; includes ZIP reading/extraction, image decoding, lazy model loading during the batch, inference, and artifact writes completed before timing.json itself is written. Network upload time before FastAPI enters the route is not included.",
        },
    )


def _model_loading_seconds(workflow_timings: dict[str, Any]) -> float:
    return sum(
        float(value)
        for key, value in workflow_timings.items()
        if key.endswith("_model_load_seconds") and isinstance(value, (int, float))
    )


def _model_loading_note(preload_enabled: bool) -> str:
    if preload_enabled:
        return (
            "PRELOAD=true loads models during application startup, before this batch "
            "handler starts. Batch total_seconds therefore measures request-time work; "
            "recorded_model_initialization_seconds reports the earlier startup loads."
        )
    return (
        "PRELOAD=false loads each required model on first use. Any model loaded during "
        "this batch is included in both the first affected item and batch total_seconds."
    )


def _validate_request(files: Sequence[UploadFile]) -> None:
    if not files:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one file is required",
        )


def _validate_expanded_batch(inputs: Sequence[BatchInput], max_files: int) -> None:
    if not inputs:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="The batch did not contain any processable files",
        )
    if len(inputs) > max_files:
        _raise_too_many(len(inputs), max_files)


def _raise_too_many(actual: int, maximum: int) -> None:
    raise HTTPException(
        status_code=413,
        detail=f"Batch contains at least {actual} files; maximum is {maximum}",
    )


async def _close_upload(file: UploadFile, operation: str) -> None:
    try:
        await file.close()
    except Exception:
        logger.warning(
            "Batch operation %s could not close uploaded file %r",
            operation,
            file.filename,
            exc_info=True,
        )
