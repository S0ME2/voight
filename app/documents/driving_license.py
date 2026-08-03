import time
from typing import Any

import cv2
import numpy as np

from app.artifacts import ArtifactWriter
from app.config import DrivingLicenseSettings
from app.documents.driving_license_fields import parse_fields, validation_warnings
from app.imaging import draw_polygon, order_corners, warp_to_size
from app.ocr import recognize
from app.roi import assign_tokens_to_rois, crop_normalized_roi, draw_roi_assignments, load_roi_config


def extract(
    image: np.ndarray,
    aligner: Any,
    ocr: Any,
    artifacts: ArtifactWriter,
    settings: DrivingLicenseSettings,
    *,
    started_total: float | None = None,
    initial_timings: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started_total = started_total if started_total is not None else time.perf_counter()
    timings: dict[str, Any] = dict(initial_timings or {})
    data_crop_roi = load_roi_config(settings.data_crop)["data_crop"]
    field_rois = load_roi_config(settings.field_rois)

    started = time.perf_counter()
    padded = cv2.copyMakeBorder(image, settings.aligner_padding, settings.aligner_padding, settings.aligner_padding, settings.aligner_padding, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    padded_corners = np.asarray(aligner(img=padded, do_center_crop=False), dtype=np.float32).reshape(4, 2)
    corners = padded_corners - settings.aligner_padding
    timings["document_detection_seconds"] = time.perf_counter() - started
    artifacts.save_image("01_docaligner_input_padded.jpg", padded)
    artifacts.save_json("02_docaligner_result.json", {"polygon_on_padded_image": padded_corners, "polygon_on_original_image": corners, "padding_pixels": settings.aligner_padding})
    artifacts.save_image("03_document_detection.jpg", draw_polygon(image, corners))

    started = time.perf_counter()
    canonical = warp_to_size(image, corners, settings.canonical_width, settings.canonical_height)
    timings["canonicalization_seconds"] = time.perf_counter() - started
    artifacts.save_image("04_canonical_license.jpg", canonical)
    started = time.perf_counter()
    data_crop = crop_normalized_roi(canonical, data_crop_roi)
    timings["data_crop_seconds"] = time.perf_counter() - started
    artifacts.save_image("05_data_crop.jpg", data_crop)

    started = time.perf_counter()
    tokens = recognize(data_crop, ocr, artifacts)
    timings["ocr_seconds"] = time.perf_counter() - started
    artifacts.save_json("08_ocr_tokens.json", tokens)
    height, width = data_crop.shape[:2]
    started = time.perf_counter()
    assignments, unassigned = assign_tokens_to_rois(tokens, field_rois, width, height, settings.min_overlap_ratio)
    timings["field_assignment_seconds"] = time.perf_counter() - started
    artifacts.save_image("09_field_assignment_annotated.jpg", draw_roi_assignments(data_crop, field_rois, assignments))

    started = time.perf_counter()
    extracted, raw_fields = parse_fields(assignments)
    timings["field_parsing_seconds"] = time.perf_counter() - started
    timings["total_seconds"] = time.perf_counter() - started_total
    ordered_corners = order_corners(corners)
    report = {
        "canonical_size": {"width": settings.canonical_width, "height": settings.canonical_height},
        "data_crop_size": {"width": width, "height": height},
        "detected_corners": {name: [float(point[0]), float(point[1])] for name, point in zip(("top_left", "top_right", "bottom_right", "bottom_left"), ordered_corners)},
        "ocr_token_count": len(tokens), "unassigned_token_count": len(unassigned), "unassigned_tokens": unassigned,
        "validation_warnings": validation_warnings(extracted), "timings": timings,
    }
    artifacts.save_json("10_raw_field_assignments.json", assignments)
    artifacts.save_json("11_raw_fields.json", raw_fields)
    artifacts.save_json("12_extracted.json", extracted)
    artifacts.save_json("13_pipeline_report.json", report)
    return extracted, report
