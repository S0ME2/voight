import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def load_roi_config(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return data


def normalized_roi_to_pixels(roi: dict[str, float], width: int, height: int) -> tuple[int, int, int, int]:
    try:
        x1, y1, x2, y2 = (float(roi[key]) for key in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid normalized ROI: {roi}") from error
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"Invalid normalized ROI: {roi}")
    left = max(0, min(width - 1, round(x1 * width)))
    top = max(0, min(height - 1, round(y1 * height)))
    right = max(left + 1, min(width, round(x2 * width)))
    bottom = max(top + 1, min(height, round(y2 * height)))
    return left, top, right, bottom


def crop_normalized_roi(image: np.ndarray, roi: dict[str, float]) -> np.ndarray:
    height, width = image.shape[:2]
    left, top, right, bottom = normalized_roi_to_pixels(roi, width, height)
    return image[top:bottom, left:right].copy()


def assign_tokens_to_rois(
    tokens: list[dict[str, Any]],
    rois: dict[str, dict[str, float]],
    width: int,
    height: int,
    min_overlap: float,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    pixel_rois = {name: normalized_roi_to_pixels(roi, width, height) for name, roi in rois.items()}
    assignments = {name: [] for name in rois}
    unassigned = []
    for token in tokens:
        left, top, right, bottom = (token[key] for key in ("x1", "y1", "x2", "y2"))
        area = max(0.0, right - left) * max(0.0, bottom - top)
        best_name, best_overlap = None, 0.0
        for name, (x1, y1, x2, y2) in pixel_rois.items():
            overlap = max(0.0, min(right, x2) - max(left, x1)) * max(0.0, min(bottom, y2) - max(top, y1))
            ratio = overlap / area if area else 0.0
            if ratio > best_overlap:
                best_name, best_overlap = name, ratio
        if best_name is None or best_overlap < min_overlap:
            unassigned.append(token)
        else:
            assignments[best_name].append({**token, "overlap_ratio": float(best_overlap)})
    return assignments, unassigned


def draw_roi_assignments(
    image: np.ndarray,
    rois: dict[str, dict[str, float]],
    assignments: dict[str, list[dict[str, Any]]],
) -> np.ndarray:
    output = image.copy()
    height, width = image.shape[:2]
    for name, roi in rois.items():
        x1, y1, x2, y2 = normalized_roi_to_pixels(roi, width, height)
        cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(output, name, (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
        for token in assignments.get(name, []):
            cv2.rectangle(output, (round(token["x1"]), round(token["y1"])), (round(token["x2"]), round(token["y2"])), (0, 0, 255), 1)
    return output
