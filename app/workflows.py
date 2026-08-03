import time
from dataclasses import dataclass
from typing import Any, Callable

from app.artifacts import ArtifactWriter, create_artifact_run, save_input
from app.config import Settings
from app.documents.driving_license import extract as extract_driving_license
from app.documents.mrz import MrzProfile, extract as extract_mrz
from app.models import Models
from app.uploads import UploadedDocument, UploadedImage


@dataclass(frozen=True)
class WorkflowResult:
    result: str | dict[str, Any]
    timings: dict[str, Any]


def inspect_file_type(
    upload: UploadedDocument,
    settings: Settings,
) -> dict[str, str | None]:
    artifacts = create_artifact_run(
        settings.artifacts,
        "file_type",
        upload.filename,
    )
    save_input(
        artifacts,
        upload.data,
        upload.filename,
        upload.content_type,
        upload.extension,
    )
    artifacts.save_json(
        "file_type_result.json",
        {"extension": upload.extension, "detected_from_file_contents": True},
    )
    return {"extension": upload.extension}


def _acquire_model(
    models: Models,
    name: str,
    getter: Callable[[], Any],
) -> tuple[Any, dict[str, Any]]:
    was_loaded = models.is_loaded(name)
    started = time.perf_counter()
    model = getter()
    access_seconds = time.perf_counter() - started
    return model, {
        f"{name}_model_access_seconds": access_seconds,
        f"{name}_model_loaded_during_request": not was_loaded,
        f"{name}_model_load_seconds": access_seconds if not was_loaded else 0.0,
        f"{name}_recorded_initial_load_seconds": models.recorded_load_seconds(name),
    }


def run_document_mrz(
    upload: UploadedImage,
    operation: str,
    profile: MrzProfile,
    models: Models,
    settings: Settings,
    *,
    artifacts: ArtifactWriter | None = None,
    save_input_artifacts: bool = True,
) -> WorkflowResult:
    started_total = time.perf_counter()
    writer = artifacts or create_artifact_run(
        settings.artifacts,
        operation,
        upload.filename,
    )
    if save_input_artifacts:
        save_input(
            writer,
            upload.data,
            upload.filename,
            upload.content_type,
            upload.extension,
            upload.image,
            source_filename=upload.source_filename,
            archive_path=upload.archive_path,
        )

    detector, detector_timing = _acquire_model(
        models,
        "mrz_scanner",
        models.mrz_scanner,
    )
    ocr, ocr_timing = _acquire_model(models, "ocr", models.ocr)
    initial_timings = {**detector_timing, **ocr_timing}
    text, timings = extract_mrz(
        upload.image,
        detector,
        ocr,
        writer,
        settings.mrz,
        profile,
        started_total=started_total,
        initial_timings=initial_timings,
    )
    return WorkflowResult(text, timings)


def extract_document_mrz(
    upload: UploadedImage,
    operation: str,
    profile: MrzProfile,
    models: Models,
    settings: Settings,
) -> str:
    """Backward-compatible single-file MRZ workflow."""

    return str(
        run_document_mrz(
            upload,
            operation,
            profile,
            models,
            settings,
        ).result
    )


def run_driving_license_fields(
    upload: UploadedImage,
    models: Models,
    settings: Settings,
    *,
    artifacts: ArtifactWriter | None = None,
    save_input_artifacts: bool = True,
) -> WorkflowResult:
    started_total = time.perf_counter()
    writer = artifacts or create_artifact_run(
        settings.artifacts,
        "driving_license",
        upload.filename,
    )
    if save_input_artifacts:
        save_input(
            writer,
            upload.data,
            upload.filename,
            upload.content_type,
            upload.extension,
            upload.image,
            source_filename=upload.source_filename,
            archive_path=upload.archive_path,
        )

    aligner, aligner_timing = _acquire_model(
        models,
        "document_aligner",
        models.document_aligner,
    )
    ocr, ocr_timing = _acquire_model(models, "ocr", models.ocr)
    initial_timings = {**aligner_timing, **ocr_timing}
    extracted, report = extract_driving_license(
        upload.image,
        aligner,
        ocr,
        writer,
        settings.driving_license,
        started_total=started_total,
        initial_timings=initial_timings,
    )
    return WorkflowResult(extracted, report["timings"])


def extract_driving_license_fields(
    upload: UploadedImage,
    models: Models,
    settings: Settings,
) -> dict:
    """Backward-compatible single-file driving-license workflow."""

    result = run_driving_license_fields(upload, models, settings).result
    if not isinstance(result, dict):
        raise TypeError("Driving-license workflow returned a non-object result")
    return result
