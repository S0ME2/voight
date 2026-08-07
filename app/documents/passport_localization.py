"""MRZ-anchored localization for the supplied Uzbekistan passport layout."""

from __future__ import annotations

from typing import Any, Callable

import cv2
import numpy as np

from app.imaging import order_corners


def page_corners_from_mrz(mrz: Any, relative_page_corners: Any) -> np.ndarray:
    """Map page corners expressed in MRZ coordinates onto one input image."""
    polygon = order_corners(np.asarray(mrz, dtype=np.float32).reshape(4, 2))
    relative = np.asarray(relative_page_corners, dtype=np.float32).reshape(4, 2)
    transform = cv2.getPerspectiveTransform(
        np.float32([[0, 0], [1, 0], [1, 1], [0, 1]]), polygon
    )
    return cv2.perspectiveTransform(relative[None], transform)[0]


def mrz_width_frame(mrz: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Return an MRZ-origin frame scaled by its stable long edge."""
    tl, tr, br, bl = order_corners(np.asarray(mrz, dtype=np.float32).reshape(4, 2))
    x_axis = tr - tl
    scale = float(np.linalg.norm(x_axis))
    if scale <= 0:
        raise ValueError("MRZ long edge has no length")
    x_axis /= scale
    y_axis = ((bl - tl) + (br - tr)) / 2
    y_axis -= x_axis * float(np.dot(y_axis, x_axis))
    length = float(np.linalg.norm(y_axis))
    if length <= 0:
        y_axis = np.float32([-x_axis[1], x_axis[0]])
    else:
        y_axis /= length
    return tl, x_axis, y_axis, scale


def relative_to_mrz_width(mrz: Any, page: Any) -> list[list[float]]:
    origin, x_axis, y_axis, scale = mrz_width_frame(mrz)
    delta = np.asarray(page, dtype=np.float32).reshape(4, 2) - origin
    return np.column_stack((delta @ x_axis / scale, delta @ y_axis / scale)).tolist()


def page_corners_from_mrz_width(mrz: Any, relative_page_corners: Any) -> np.ndarray:
    origin, x_axis, y_axis, scale = mrz_width_frame(mrz)
    relative = np.asarray(relative_page_corners, dtype=np.float32).reshape(4, 2)
    return origin + scale * (relative[:, :1] * x_axis + relative[:, 1:] * y_axis)


def detect_passport_page(
    image: np.ndarray,
    profile: dict[str, Any],
    mrz_detector: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    localization = profile.get("document_localization", {})
    if localization.get("strategy") != "mrz_anchor":
        raise ValueError("passport profile requires MRZ-anchored localization")
    detected = mrz_detector(image, do_center_crop=False)
    mrz = np.asarray(detected["mrz_polygon"], dtype=np.float32).reshape(4, 2)
    return {
        "corners": page_corners_from_mrz_width(mrz, localization["page_corners_relative_to_mrz_width"]),
        "mrz_polygon": mrz,
    }


def detect_passport_page_padded(
    padded_image: np.ndarray,
    profile: dict[str, Any],
    mrz_detector: Callable[..., dict[str, Any]],
    padding: int,
) -> dict[str, Any]:
    """Keep the MRZ detector in the original-image coordinate frame."""
    image = padded_image[padding:-padding, padding:-padding] if padding else padded_image
    result = detect_passport_page(image, profile, mrz_detector)
    result["corners"] = np.asarray(result["corners"], dtype=np.float32) + padding
    return result
