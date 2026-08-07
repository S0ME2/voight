"""Model-agnostic profile extraction mechanics."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from app.artifacts import ArtifactWriter
from app.imaging import draw_polygon, order_corners, warp_to_size
from app.roi import (
    assign_tokens_to_rois,
    crop_normalized_roi,
    draw_roi_assignments,
    load_roi_config,
    normalized_roi_to_pixels,
)

Token = dict[str, Any]


@dataclass(frozen=True)
class RegionProfile:
    data_crop: dict[str, float]
    field_rois: dict[str, dict[str, float]]


@dataclass
class PreparedProfile:
    image: np.ndarray
    profile: RegionProfile
    artifacts: ArtifactWriter
    canonical_width: int
    canonical_height: int
    crop_bounds: tuple[int, int, int, int]
    data_crop: np.ndarray
    document_confidence: dict[str, Any] | None
    corners: np.ndarray
    timings: dict[str, Any]
    started_total: float


@lru_cache(maxsize=None)
def load_region_profile(data_crop_path: Path, field_rois_path: Path) -> RegionProfile:
    """Load and validate reusable profile geometry once per process."""
    data_crop = load_roi_config(data_crop_path).get("data_crop")
    field_rois = load_roi_config(field_rois_path)
    normalized_roi_to_pixels(data_crop, 2, 2)
    if not field_rois:
        raise ValueError("A profile requires at least one field ROI")
    for roi in field_rois.values():
        normalized_roi_to_pixels(roi, 2, 2)
    return RegionProfile(data_crop, field_rois)


def detect_document_corners(
    image: np.ndarray,
    detect_document: Callable[[np.ndarray], Any],
    padding: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any] | None]:
    padded = cv2.copyMakeBorder(
        image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0, 0, 0)
    )
    detected = detect_document(padded)
    confidence = None
    if isinstance(detected, dict):
        corners_value = detected.get("corners")
        score = detected.get("score")
        if score is not None:
            confidence = {
                "score": float(score),
                "source": "document_detection",
                "calibrated_probability": False,
            }
    else:
        corners_value = detected
    padded_corners = np.asarray(corners_value, dtype=np.float32).reshape(4, 2)
    corners = padded_corners - padding
    return padded, padded_corners, corners, confidence


def rectify_document(
    image: np.ndarray, corners: np.ndarray, width: int, height: int
) -> np.ndarray:
    return warp_to_size(image, corners, width, height)


def _canonical_box(
    token: Token,
    crop_bounds: tuple[int, int, int, int],
    canonical_width: int,
    canonical_height: int,
) -> dict[str, float]:
    crop_left, crop_top, _, _ = crop_bounds
    return {
        "x1": max(0.0, min(1.0, (crop_left + float(token["x1"])) / canonical_width)),
        "y1": max(0.0, min(1.0, (crop_top + float(token["y1"])) / canonical_height)),
        "x2": max(0.0, min(1.0, (crop_left + float(token["x2"])) / canonical_width)),
        "y2": max(0.0, min(1.0, (crop_top + float(token["y2"])) / canonical_height)),
    }


def add_canonical_boxes(
    tokens: list[Token],
    crop_bounds: tuple[int, int, int, int],
    canonical_width: int,
    canonical_height: int,
) -> list[Token]:
    return [
        {
            **token,
            "canonical_box": _canonical_box(
                token, crop_bounds, canonical_width, canonical_height
            ),
        }
        for token in tokens
    ]


def aggregate_field_evidence(
    assignments: dict[str, list[Token]],
) -> tuple[dict[str, dict[str, Any] | None], dict[str, dict[str, float] | None]]:
    """Return mean OCR score and union box for every configured field."""
    confidences: dict[str, dict[str, Any] | None] = {}
    boxes: dict[str, dict[str, float] | None] = {}
    for field, tokens in assignments.items():
        scored = [float(token["score"]) for token in tokens if token.get("score") is not None]
        confidences[field] = (
            {
                "score": sum(scored) / len(scored),
                "source": "ocr_token_mean",
                "calibrated_probability": False,
            }
            if scored
            else None
        )
        token_boxes = [token["canonical_box"] for token in tokens if "canonical_box" in token]
        boxes[field] = (
            {
                "x1": min(box["x1"] for box in token_boxes),
                "y1": min(box["y1"] for box in token_boxes),
                "x2": max(box["x2"] for box in token_boxes),
                "y2": max(box["y2"] for box in token_boxes),
            }
            if token_boxes
            else None
        )
    return confidences, boxes


def record_rectification_artifacts(
    artifacts: ArtifactWriter,
    image: np.ndarray,
    padded: np.ndarray,
    padded_corners: np.ndarray,
    corners: np.ndarray,
    padding: int,
    canonical: np.ndarray,
    data_crop: np.ndarray,
) -> None:
    artifacts.save_image("01_docaligner_input_padded.jpg", padded)
    artifacts.save_json(
        "02_docaligner_result.json",
        {
            "polygon_on_padded_image": padded_corners,
            "polygon_on_original_image": corners,
            "padding_pixels": padding,
        },
    )
    artifacts.save_image("03_document_detection.jpg", draw_polygon(image, corners))
    artifacts.save_image("04_canonical_license.jpg", canonical)
    artifacts.save_image("05_data_crop.jpg", data_crop)


def record_extraction_artifacts(
    artifacts: ArtifactWriter,
    data_crop: np.ndarray,
    field_rois: dict[str, dict[str, float]],
    tokens: list[Token],
    assignments: dict[str, list[Token]],
    raw_fields: dict[str, str],
    extracted: dict[str, Any],
    report: dict[str, Any],
) -> None:
    artifacts.save_json("08_ocr_tokens.json", tokens)
    artifacts.save_image(
        "09_field_assignment_annotated.jpg",
        draw_roi_assignments(data_crop, field_rois, assignments),
    )
    artifacts.save_json("10_raw_field_assignments.json", assignments)
    artifacts.save_json("11_raw_fields.json", raw_fields)
    artifacts.save_json("12_extracted.json", extracted)
    artifacts.save_json("13_pipeline_report.json", report)


def prepare_profile(
    image: np.ndarray,
    profile: RegionProfile,
    detect_document: Callable[[np.ndarray], Any],
    artifacts: ArtifactWriter,
    *,
    canonical_width: int,
    canonical_height: int,
    padding: int,
    started_total: float | None = None,
    initial_timings: dict[str, Any] | None = None,
) -> PreparedProfile:
    """Localize and crop one profile before shared batched OCR."""
    started_total = started_total if started_total is not None else time.perf_counter()
    timings: dict[str, Any] = dict(initial_timings or {})

    started = time.perf_counter()
    padded, padded_corners, corners, document_confidence = detect_document_corners(
        image, detect_document, padding
    )
    timings["document_detection_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    canonical = rectify_document(image, corners, canonical_width, canonical_height)
    timings["canonicalization_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    crop_bounds = normalized_roi_to_pixels(
        profile.data_crop, canonical_width, canonical_height
    )
    data_crop = crop_normalized_roi(canonical, profile.data_crop)
    timings["data_crop_seconds"] = time.perf_counter() - started
    record_rectification_artifacts(
        artifacts, image, padded, padded_corners, corners, padding, canonical, data_crop
    )
    return PreparedProfile(
        image=image,
        profile=profile,
        artifacts=artifacts,
        canonical_width=canonical_width,
        canonical_height=canonical_height,
        crop_bounds=crop_bounds,
        data_crop=data_crop,
        document_confidence=document_confidence,
        corners=corners,
        timings=timings,
        started_total=started_total,
    )


def complete_profile(
    prepared: PreparedProfile,
    tokens: list[Token],
    parse_fields: Callable[[dict[str, list[Token]]], tuple[dict[str, Any], dict[str, str]]],
    validate_fields: Callable[[dict[str, Any]], list[str]],
    *,
    min_overlap: float,
    ocr_seconds: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Restore batched OCR tokens and finish one profile independently."""
    timings = prepared.timings
    timings["ocr_seconds"] = ocr_seconds
    tokens = add_canonical_boxes(
        tokens,
        prepared.crop_bounds,
        prepared.canonical_width,
        prepared.canonical_height,
    )

    height, width = prepared.data_crop.shape[:2]
    started = time.perf_counter()
    assignments, unassigned = assign_tokens_to_rois(
        tokens, prepared.profile.field_rois, width, height, min_overlap
    )
    timings["field_assignment_seconds"] = time.perf_counter() - started

    started = time.perf_counter()
    extracted, raw_fields = parse_fields(assignments)
    timings["field_parsing_seconds"] = time.perf_counter() - started
    confidences, boxes = aggregate_field_evidence(assignments)
    started = time.perf_counter()
    warnings = validate_fields(extracted)
    timings["validation_seconds"] = time.perf_counter() - started
    timings["total_seconds"] = time.perf_counter() - prepared.started_total
    ordered_corners = order_corners(prepared.corners)
    report = {
        "canonical_size": {
            "width": prepared.canonical_width,
            "height": prepared.canonical_height,
        },
        "data_crop_size": {"width": width, "height": height},
        "detected_corners": {
            name: [float(point[0]), float(point[1])]
            for name, point in zip(
                ("top_left", "top_right", "bottom_right", "bottom_left"),
                ordered_corners,
            )
        },
        "ocr_token_count": len(tokens),
        "unassigned_token_count": len(unassigned),
        "unassigned_tokens": unassigned,
        "document_confidence": prepared.document_confidence,
        "raw_fields": raw_fields,
        "field_raw_text": {
            field: [token["text"] for token in field_tokens if token.get("text")]
            for field, field_tokens in assignments.items()
        },
        "field_confidences": confidences,
        "field_bounding_boxes": boxes,
        "validation_warnings": warnings,
        "timings": timings,
    }
    record_extraction_artifacts(
        prepared.artifacts,
        prepared.data_crop,
        prepared.profile.field_rois,
        tokens,
        assignments,
        raw_fields,
        extracted,
        report,
    )
    return extracted, report


def extract_profile(
    image: np.ndarray,
    profile: RegionProfile,
    detect_document: Callable[[np.ndarray], Any],
    recognize_tokens: Callable[[np.ndarray], list[Token]],
    parse_fields: Callable[[dict[str, list[Token]]], tuple[dict[str, Any], dict[str, str]]],
    validate_fields: Callable[[dict[str, Any]], list[str]],
    artifacts: ArtifactWriter,
    *,
    canonical_width: int,
    canonical_height: int,
    padding: int,
    min_overlap: float,
    started_total: float | None = None,
    initial_timings: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Run shared geometry and assignment with injected model callables."""
    prepared = prepare_profile(
        image,
        profile,
        detect_document,
        artifacts,
        canonical_width=canonical_width,
        canonical_height=canonical_height,
        padding=padding,
        started_total=started_total,
        initial_timings=initial_timings,
    )
    started = time.perf_counter()
    tokens = recognize_tokens(prepared.data_crop)
    return complete_profile(
        prepared,
        tokens,
        parse_fields,
        validate_fields,
        min_overlap=min_overlap,
        ocr_seconds=time.perf_counter() - started,
    )
