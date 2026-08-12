"""Profile-driven Uzbekistan passport and paired ID-card extraction."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from app.artifacts import ArtifactWriter, create_child_artifact_run
from app.contracts import (
    Confidence,
    DocumentResult,
    DocumentType,
    ErrorCode,
    ErrorResult,
    FieldResult,
    TimingResult,
    ValidationResult,
    ValidationStatus,
)
from app.documents.mrz import ID_CARD, PASSPORT, MrzProfile, parse as parse_mrz
from app.documents.profiles import load_document_profile
from app.pipeline import RegionProfile, extract_profile

Token = dict[str, Any]


@dataclass(frozen=True)
class IdentityBatchPipeline:
    regions: tuple[str, ...]
    localization_kind: str
    mrz_region: str
    mrz_profile: MrzProfile
    reconcile_mapping: dict[str, str]

    def region_profile(self, profile: dict[str, Any], region: str) -> RegionProfile:
        geometry = profile["regions"][region]
        return RegionProfile(geometry["data_crop"], geometry["field_rois"])


def passport_batch_pipeline() -> IdentityBatchPipeline:
    return IdentityBatchPipeline(
        ("data_page",),
        "mrz",
        "data_page",
        PASSPORT,
        {
            "surname": "surname",
            "name": "given_names",
            "passport_number": "document_number",
            "date_of_birth": "date_of_birth",
            "sex": "sex",
            "date_of_expiry": "date_of_expiry",
        },
    )


def id_card_batch_pipeline() -> IdentityBatchPipeline:
    return IdentityBatchPipeline(
        ("front", "back"),
        "docaligner",
        "back",
        ID_CARD,
        {
            "surname": "surname",
            "name": "given_names",
            "card_number": "document_number",
            "pinfl": "pinfl",
            "date_of_birth": "date_of_birth",
            "sex": "sex",
            "citizenship": "nationality",
            "date_of_expiry": "date_of_expiry",
        },
    )


class DocumentPipelineError(Exception):
    """A client-safe item failure for the API/batch layer."""

    def __init__(self, code: ErrorCode, detail: str):
        super().__init__(detail)
        self.error = ErrorResult(code=code, detail=detail)


def _ordered_text(tokens: list[Token]) -> list[str]:
    return [
        str(token["text"]).strip()
        for token in sorted(tokens, key=lambda token: (token.get("center_y", 0), token.get("x1", 0)))
        if str(token.get("text", "")).strip()
    ]


def _parse_visible(assignments: dict[str, list[Token]]) -> tuple[dict[str, str | None], dict[str, str]]:
    raw = {field: " ".join(_ordered_text(tokens)) for field, tokens in assignments.items()}
    return ({field: " ".join(value.split()) or None for field, value in raw.items()}, raw)


def _required_warnings(profile: dict[str, Any], region: str) -> Callable[[dict[str, Any]], list[str]]:
    required = [
        field["name"]
        for field in profile["fields"]
        if field["region"] == region and field["required"]
    ]
    return lambda values: [
        f"Missing required field: {field}" for field in required if not values.get(field)
    ]


def _extract_region(
    image: np.ndarray,
    profile: dict[str, Any],
    region: str,
    detect_document: Callable[[np.ndarray], Any],
    recognize_tokens: Callable[[np.ndarray], list[Token]],
    artifacts: ArtifactWriter,
    *,
    padding: int,
    min_overlap: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    geometry = profile["regions"][region]
    width = int(profile["canonical_size"]["width"])
    height = int(profile["canonical_size"]["height"])
    try:
        return extract_profile(
            image,
            RegionProfile(geometry["data_crop"], geometry["field_rois"]),
            detect_document,
            recognize_tokens,
            _parse_visible,
            _required_warnings(profile, region),
            artifacts,
            canonical_width=width,
            canonical_height=height,
            padding=padding,
            min_overlap=min_overlap,
        )
    except DocumentPipelineError:
        raise
    except (cv2.error, IndexError, KeyError, TypeError, ValueError) as error:
        raise DocumentPipelineError(
            ErrorCode.INVALID_DOCUMENT, f"{region} image could not be localized"
        ) from error


def _field_results(
    profile: dict[str, Any],
    region: str,
    values: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, FieldResult]:
    return {
        field["name"]: FieldResult(
            value=values.get(field["name"]),
            raw_text=report["field_raw_text"][field["name"]],
            confidence=report["field_confidences"][field["name"]],
            bounding_box=report["field_bounding_boxes"][field["name"]],
            region=region,
        )
        for field in profile["fields"]
        if field["region"] == region
    }


def _mrz(read_mrz: Callable[[np.ndarray], str], image: np.ndarray, document_type: str):
    try:
        return parse_mrz(read_mrz(image) or "", document_type)
    except (cv2.error, IndexError, KeyError, TypeError, ValueError):
        return parse_mrz("", document_type)


def _date_key(visible: str) -> str:
    digits = re.sub(r"\D", "", visible)
    return digits[6:8] + digits[2:4] + digits[0:2] if len(digits) == 8 else digits


def _key(field: str, value: str) -> str:
    value = value.upper()
    if field in {"date_of_birth", "date_of_expiry"}:
        return _date_key(value)
    if field == "sex":
        return {"AYOL": "F", "ERKAK": "M"}.get(value, value)
    if field in {"citizenship", "nationality"} and "ZBEK" in value:
        return "UZB"
    return re.sub(r"[^A-Z0-9]", "", value)


def _reconcile(
    visible: dict[str, FieldResult],
    mrz_fields: dict[str, str | None],
    mapping: dict[str, str],
) -> list[ValidationResult]:
    validations = []
    for visible_name, mrz_name in mapping.items():
        visible_value = visible.get(visible_name).value if visible_name in visible else None
        mrz_value = mrz_fields.get(mrz_name)
        if visible_value is None or mrz_value is None:
            status, detail = ValidationStatus.NOT_RUN, "field is not present in both sources"
        elif _key(visible_name, visible_value) == _key(mrz_name, mrz_value):
            status, detail = ValidationStatus.PASSED, None
        else:
            status = ValidationStatus.FAILED
            detail = f"visible text {visible_value!r} conflicts with MRZ {mrz_value!r}"
        validations.append(
            ValidationResult(
                code=f"visible_mrz_{visible_name}",
                status=status,
                detail=detail,
                fields=[visible_name, f"mrz.{mrz_name}"],
            )
        )
    return validations


def _document_confidence(reports: list[dict[str, Any]]) -> Confidence | None:
    confidences = [report.get("document_confidence") for report in reports]
    if not confidences or any(confidence is None for confidence in confidences):
        return None
    return Confidence(
        score=min(confidence["score"] for confidence in confidences),
        source="document_detection",
        calibrated_probability=False,
    )


def _required_validation(profile: dict[str, Any], fields: dict[str, FieldResult]) -> ValidationResult:
    missing = [
        field["name"]
        for field in profile["fields"]
        if field["required"] and fields[field["name"]].value is None
    ]
    return ValidationResult(
        code="required_visible_fields",
        status=ValidationStatus.WARNING if missing else ValidationStatus.PASSED,
        detail=f"missing: {', '.join(missing)}" if missing else None,
        fields=missing,
    )


def extract_passport(
    image: np.ndarray | None,
    profile_path: Path,
    detect_document: Callable[[np.ndarray], Any],
    recognize_tokens: Callable[[np.ndarray], list[Token]],
    read_mrz: Callable[[np.ndarray], str],
    artifacts: ArtifactWriter,
    *,
    padding: int = 0,
    min_overlap: float = 0.3,
) -> DocumentResult:
    if image is None:
        raise DocumentPipelineError(ErrorCode.INVALID_UPLOAD, "passport image is required")
    started = time.perf_counter()
    try:
        profile = load_document_profile(profile_path)
    except ValueError as error:
        raise DocumentPipelineError(ErrorCode.PROFILE_UNAVAILABLE, "passport profile is unavailable") from error
    values, report = _extract_region(
        image, profile, "data_page", detect_document, recognize_tokens, artifacts,
        padding=padding, min_overlap=min_overlap,
    )
    fields = _field_results(profile, "data_page", values, report)
    mrz = _mrz(read_mrz, image, "passport")
    validations = [_required_validation(profile, fields)] + _reconcile(
        fields,
        mrz.fields,
        {
            "surname": "surname",
            "name": "given_names",
            "passport_number": "document_number",
            "date_of_birth": "date_of_birth",
            "sex": "sex",
            "date_of_expiry": "date_of_expiry",
        },
    )
    warnings = list(report["validation_warnings"])
    if not mrz.raw_lines:
        warnings.append("MRZ was not found")
    elif any(validation.status == ValidationStatus.FAILED for validation in mrz.validations):
        warnings.append("MRZ check-digit validation failed")
    result = DocumentResult(
        document_type=DocumentType.PASSPORT,
        layout=profile["layout"],
        fields=fields,
        document_confidence=_document_confidence([report]),
        mrz=mrz,
        validations=validations,
        warnings=warnings,
        timings=TimingResult(total_seconds=time.perf_counter() - started, stages=report["timings"]),
    )
    artifacts.save_json("14_document_result.json", result.model_dump(mode="json"))
    return result


def extract_id_card(
    front: np.ndarray | None,
    back: np.ndarray | None,
    profile_path: Path,
    detect_document: Callable[[np.ndarray], Any],
    recognize_tokens: Callable[[np.ndarray], list[Token]],
    read_mrz: Callable[[np.ndarray], str],
    artifacts: ArtifactWriter,
    *,
    padding: int = 0,
    min_overlap: float = 0.3,
) -> DocumentResult:
    if front is None or back is None:
        raise DocumentPipelineError(
            ErrorCode.MISSING_DOCUMENT_SIDE,
            "id_card requires both front and back images",
        )
    started = time.perf_counter()
    try:
        profile = load_document_profile(profile_path)
    except ValueError as error:
        raise DocumentPipelineError(ErrorCode.PROFILE_UNAVAILABLE, "ID-card profile is unavailable") from error
    front_mrz = _mrz(read_mrz, front, "id_card")
    back_mrz = _mrz(read_mrz, back, "id_card")
    if front_mrz.raw_lines and not back_mrz.raw_lines:
        raise DocumentPipelineError(
            ErrorCode.INVALID_DOCUMENT,
            "ID-card front and back appear to be swapped",
        )
    front_values, front_report = _extract_region(
        front, profile, "front", detect_document, recognize_tokens,
        create_child_artifact_run(artifacts, 0, "front"), padding=padding, min_overlap=min_overlap,
    )
    back_values, back_report = _extract_region(
        back, profile, "back", detect_document, recognize_tokens,
        create_child_artifact_run(artifacts, 1, "back"), padding=padding, min_overlap=min_overlap,
    )
    fields = _field_results(profile, "front", front_values, front_report)
    back_fields = _field_results(profile, "back", back_values, back_report)
    conflicts = fields.keys() & back_fields.keys()
    if conflicts:
        raise DocumentPipelineError(
            ErrorCode.PROCESSING_FAILED,
            f"duplicate ID-card field ownership: {', '.join(sorted(conflicts))}",
        )
    fields.update(back_fields)
    validations = [_required_validation(profile, fields)] + _reconcile(
        fields,
        back_mrz.fields,
        {
            "surname": "surname",
            "name": "given_names",
            "card_number": "document_number",
            "pinfl": "pinfl",
            "date_of_birth": "date_of_birth",
            "sex": "sex",
            "citizenship": "nationality",
            "date_of_expiry": "date_of_expiry",
        },
    )
    warnings = front_report["validation_warnings"] + back_report["validation_warnings"]
    if not back_mrz.raw_lines:
        warnings.append("MRZ was not found on ID-card back")
    elif any(validation.status == ValidationStatus.FAILED for validation in back_mrz.validations):
        warnings.append("MRZ check-digit validation failed")
    stages = {
        **{f"front_{name}": value for name, value in front_report["timings"].items()},
        **{f"back_{name}": value for name, value in back_report["timings"].items()},
    }
    result = DocumentResult(
        document_type=DocumentType.ID_CARD,
        layout=profile["layout"],
        fields=fields,
        document_confidence=_document_confidence([front_report, back_report]),
        mrz=back_mrz,
        validations=validations,
        warnings=warnings,
        timings=TimingResult(total_seconds=time.perf_counter() - started, stages=stages),
    )
    artifacts.save_json("id_card_document_result.json", result.model_dump(mode="json"))
    return result
