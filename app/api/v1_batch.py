"""Typed job construction and result assembly for the frozen v1 transport."""

from __future__ import annotations

import time
import resource
from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

from app.api.schemas import OcrBatchResponse
from app.artifacts import ArtifactWriter, create_child_artifact_run, save_input
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
from app.documents.driving_license_fields import split_birth_line, validation_warnings
from app.documents.identity import (
    DocumentPipelineError,
    document_confidence,
    field_results,
    identity_batch_pipeline,
    parse_visible,
    reconcile,
    required_validation,
    required_warnings,
)
from app.documents.mrz import parse as parse_mrz
from app.documents.profiles import load_document_profile
from app.inference import ProfileBatchItem
from app.inference.batch import ProfileBatchOutcome
from app.pipeline import load_region_profile
from app.uploads import UploadedDocument, image_from_document


@dataclass(frozen=True)
class LogicalInput:
    input: DocumentInput
    files: tuple[UploadedDocument, ...]


@dataclass(frozen=True)
class BatchOwner:
    index: int
    region: str
    profile: dict[str, Any] | None
    document_type: DocumentType


@dataclass(frozen=True)
class BatchPlan:
    jobs: tuple[ProfileBatchItem, ...]
    owners: tuple[BatchOwner, ...]
    item_errors: dict[int, ErrorResult]


def driving_field_results(values: dict[str, Any], report: dict[str, Any]) -> dict[str, FieldResult]:
    source_names = {
        "surname": ("surname",),
        "given_names": ("given_names", "name"),
        "patronymic": ("patronymic",),
        "birth_place": ("birth_place", "place_of_birth"),
        "birth_date": ("birth_date", "date_of_birth", "birth_place", "place_of_birth"),
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
            value = report[kind].get(source)
            if value:
                return value
        return [] if kind == "field_raw_text" else None

    def raw_text(name: str) -> list[str]:
        values = evidence(name, "field_raw_text")
        if name == "birth_date" and values:
            for source in source_names[name]:
                if report["field_raw_text"].get(source):
                    if source == "date_of_birth":
                        return values
                    return [date or text for text in values for _, date in [split_birth_line(text)]]
        if name == "birth_place" and values:
            return [split_birth_line(text)[0] for text in values]
        return values

    return {
        name: FieldResult(
            value=value,
            raw_text=raw_text(name),
            confidence=evidence(name, "field_confidences"),
            bounding_box=evidence(name, "field_bounding_boxes"),
            region="image",
        )
        for name, value in values.items()
    }


def build_batch_plan(
    inputs: list[LogicalInput], settings: Settings, artifacts: ArtifactWriter
) -> BatchPlan:
    jobs: list[ProfileBatchItem] = []
    owners: list[BatchOwner] = []
    item_errors: dict[int, ErrorResult] = {}
    for index, logical in enumerate(inputs):
        try:
            decoded_files = tuple(image_from_document(file) for file in logical.files)
        except HTTPException:
            item_errors[index] = ErrorResult(code=ErrorCode.INVALID_UPLOAD, detail="Upload must be a decodable image")
            continue
        if logical.input.document_type == DocumentType.DRIVING_LICENSE:
            profile = load_region_profile(settings.driving_license.data_crop, settings.driving_license.field_rois)
            document_profile = None
            pipeline = None
            regions = [
                (
                    "image",
                    decoded_files[0],
                    profile,
                    settings.driving_license.canonical_width,
                    settings.driving_license.canonical_height,
                    parse_license_fields,
                    validation_warnings,
                )
            ]
        else:
            profile_path = settings.profiles.passport if logical.input.document_type == DocumentType.PASSPORT else settings.profiles.id_card
            document_profile = load_document_profile(profile_path)
            pipeline = identity_batch_pipeline(logical.input.document_type)
            regions = [
                (
                    region,
                    file,
                    pipeline.region_profile(document_profile, region),
                    int(document_profile["canonical_size"]["width"]),
                    int(document_profile["canonical_size"]["height"]),
                    parse_visible,
                    required_warnings(document_profile, region),
                )
                for region, file in zip(pipeline.regions, decoded_files)
            ]
        for region, file, profile, width, height, parser, validator in regions:
            image = file.image
            writer = create_child_artifact_run(artifacts, len(jobs), file.filename)
            save_input(
                writer,
                file.data,
                file.filename,
                file.content_type,
                file.extension,
                image,
                source_filename=file.source_filename,
                archive_path=file.archive_path,
            )
            jobs.append(
                ProfileBatchItem(
                    item_id=f"{index}:{region}",
                    image=image,
                    profile=profile,
                    localization_kind="docaligner" if pipeline is None else pipeline.localization_kind,
                    parse_fields=parser,
                    validate_fields=validator,
                    artifacts=writer,
                    canonical_width=width,
                    canonical_height=height,
                    padding=settings.driving_license.aligner_padding,
                    min_overlap=settings.driving_license.min_overlap_ratio,
                    passport_page_corners=(
                        document_profile["document_localization"]["page_corners_relative_to_mrz_width"]
                        if logical.input.document_type == DocumentType.PASSPORT
                        else None
                    ),
                    mrz_profile=pipeline.mrz_profile if pipeline and region == pipeline.mrz_region else None,
                    probe_mrz=logical.input.document_type == DocumentType.ID_CARD and region == "back",
                    mrz_fallback_for=(
                        f"{index}:back"
                        if logical.input.document_type == DocumentType.ID_CARD and region == "front"
                        else None
                    ),
                    document_id=logical.input.image or logical.input.front or str(index),
                )
            )
            owners.append(BatchOwner(index, region, document_profile, logical.input.document_type))
    return BatchPlan(tuple(jobs), tuple(owners), item_errors)


def item_error(index: int, source: DocumentInput, error: ErrorResult) -> BatchItemResult:
    return BatchItemResult(index=index, input=source, success=False, error=error)


def assemble_batch_response(
    inputs: list[LogicalInput],
    plan: BatchPlan,
    outcomes: list[ProfileBatchOutcome],
    diagnostics: dict[str, Any],
    started: float,
) -> OcrBatchResponse:
    profile_parse_validation_seconds = sum(
        sum(
            float(outcome.result[1]["timings"].get(name, 0.0))
            for name in ("field_assignment_seconds", "field_parsing_seconds", "validation_seconds")
        )
        for outcome in outcomes
        if outcome.result is not None
    )
    mrz_parse_validation_seconds = 0.0
    grouped: dict[int, dict[str, tuple[ProfileBatchOutcome, dict[str, Any] | None]]] = {}
    for owner, outcome in zip(plan.owners, outcomes):
        grouped.setdefault(owner.index, {})[owner.region] = (outcome, owner.profile)
    items: list[BatchItemResult] = []
    for index, logical in enumerate(inputs):
        if index in plan.item_errors:
            items.append(item_error(index, logical.input, plan.item_errors[index]))
            continue
        regions = grouped.get(index, {})
        if logical.input.document_type == DocumentType.ID_CARD:
            front = regions.get("front", (None, None))[0]
            back = regions.get("back", (None, None))[0]
            if front and back and front.mrz_detected and not back.mrz_detected:
                items.append(item_error(index, logical.input, ErrorResult(code=ErrorCode.INVALID_DOCUMENT, detail="ID-card front and back appear to be swapped")))
                continue
        failed = next((outcome.error for outcome, _ in regions.values() if outcome.error), None)
        if failed:
            code = failed.error.code if isinstance(failed, DocumentPipelineError) else ErrorCode.INVALID_DOCUMENT
            detail = failed.error.detail if isinstance(failed, DocumentPipelineError) else "document could not be processed"
            items.append(item_error(index, logical.input, ErrorResult(code=code, detail=detail)))
            continue
        if logical.input.document_type == DocumentType.DRIVING_LICENSE:
            values, report = regions["image"][0].result
            fields = driving_field_results(values, report)
            result = DocumentResult(
                document_type=DocumentType.DRIVING_LICENSE,
                layout="driving_license",
                fields=fields,
                warnings=report["validation_warnings"],
                timings=TimingResult(total_seconds=report["timings"]["total_seconds"], stages=report["timings"]),
            )
        else:
            profile = next(profile for _, profile in regions.values() if profile is not None)
            fields: dict[str, FieldResult] = {}
            warnings: list[str] = []
            for region, (outcome, _) in regions.items():
                values, report = outcome.result
                fields.update(field_results(profile, region, values, report))
                warnings.extend(report["validation_warnings"])
            pipeline = identity_batch_pipeline(logical.input.document_type)
            parse_started = time.perf_counter()
            mrz = parse_mrz(regions[pipeline.mrz_region][0].mrz_text or "", logical.input.document_type.value)
            validations = [required_validation(profile, fields)] + reconcile(fields, mrz.fields, pipeline.reconcile_mapping)
            mrz_parse_validation_seconds += time.perf_counter() - parse_started
            if not mrz.raw_lines:
                warnings.append("MRZ was not found" + (" on ID-card back" if logical.input.document_type == DocumentType.ID_CARD else ""))
            elif any(validation.status.value == "failed" for validation in mrz.validations):
                warnings.append("MRZ check-digit validation failed")
            reports = [outcome.result[1] for outcome, _ in regions.values()]
            stages = {
                f"{region}_{name}": value
                for region, (outcome, _) in regions.items()
                for name, value in outcome.result[1]["timings"].items()
            }
            result = DocumentResult(
                document_type=logical.input.document_type,
                layout=profile["layout"],
                fields=fields,
                document_confidence=document_confidence(reports),
                mrz=mrz,
                validations=validations,
                warnings=warnings,
                timings=TimingResult(
                    total_seconds=max((report["timings"]["total_seconds"] for report in reports), default=0.0),
                    stages=stages,
                ),
            )
        items.append(BatchItemResult(index=index, input=logical.input, success=True, result=result))
    diagnostics.setdefault("pipeline", {})["parsing_validation_seconds"] = profile_parse_validation_seconds + mrz_parse_validation_seconds
    diagnostics["pipeline"]["profile_parse_validation_seconds"] = profile_parse_validation_seconds
    diagnostics["pipeline"]["mrz_parse_validation_seconds"] = mrz_parse_validation_seconds
    diagnostics["process_peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    return OcrBatchResponse(
        total=len(items),
        succeeded=sum(item.success for item in items),
        failed=sum(not item.success for item in items),
        total_seconds=time.perf_counter() - started,
        items=items,
        diagnostics=diagnostics,
    )
