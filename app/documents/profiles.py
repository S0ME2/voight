"""Validation and loading for versioned document extraction profiles."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any


def _rectangle(value: Any, label: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    try:
        x1, y1, x2, y2 = (float(value[name]) for name in ("x1", "y1", "x2", "y2"))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{label} must contain numeric x1, y1, x2, y2") from error
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"{label} must be normalized bounds within [0, 1]")


@lru_cache(maxsize=None)
def load_document_profile(path: Path) -> dict[str, Any]:
    """Load one profile and reject geometry or field ownership mistakes."""
    try:
        profile = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Cannot load profile {path}: {error}") from error
    if not isinstance(profile, dict) or profile.get("version") != 1 or not isinstance(profile.get("layout"), str):
        raise ValueError("profile requires version 1 and a layout")
    if not isinstance(profile.get("sample_metadata"), dict):
        raise ValueError("profile requires sample_metadata")
    canonical_size = profile.get("canonical_size")
    if (
        not isinstance(canonical_size, dict)
        or not isinstance(canonical_size.get("width"), int)
        or not isinstance(canonical_size.get("height"), int)
        or canonical_size["width"] <= 0
        or canonical_size["height"] <= 0
    ):
        raise ValueError("profile requires a positive integer canonical_size")
    regions = profile.get("regions")
    fields = profile.get("fields")
    if not isinstance(regions, dict) or not regions or not isinstance(fields, list) or not fields:
        raise ValueError("profile requires non-empty regions and fields")
    roi_names: set[tuple[str, str]] = set()
    for region_name, region in regions.items():
        if not isinstance(region_name, str) or not region_name:
            raise ValueError("region names must be non-empty strings")
        _rectangle(region.get("data_crop") if isinstance(region, dict) else None, f"region {region_name} data_crop")
        rois = region.get("field_rois") if isinstance(region, dict) else None
        if not isinstance(rois, dict):
            raise ValueError(f"region {region_name} requires field_rois")
        for name, roi in rois.items():
            if not isinstance(name, str) or not name:
                raise ValueError(f"region {region_name} has an invalid field name")
            _rectangle(roi, f"region {region_name} field {name}")
            roi_names.add((region_name, name))
    names: set[str] = set()
    for field in fields:
        if not isinstance(field, dict) or not isinstance(field.get("name"), str) or not field["name"]:
            raise ValueError("each field requires a non-empty name")
        if field["name"] in names:
            raise ValueError(f"duplicate field ownership: {field['name']}")
        if not isinstance(field.get("required"), bool):
            raise ValueError(f"field {field['name']} requires a boolean required flag")
        owner = (field.get("region"), field["name"])
        if owner not in roi_names:
            raise ValueError(f"field {field['name']} is missing from its region ROI map")
        names.add(field["name"])
    if roi_names != {(field["region"], field["name"]) for field in fields}:
        raise ValueError("every ROI must have exactly one field owner")
    localization = profile.get("document_localization")
    if localization is not None:
        corners = localization.get("page_corners_relative_to_mrz_width") if isinstance(localization, dict) else None
        if localization.get("strategy") != "mrz_anchor" or not isinstance(corners, list) or len(corners) != 4:
            raise ValueError("document_localization requires an MRZ page anchor")
        if any(not isinstance(corner, list) or len(corner) != 2 or any(not isinstance(value, (int, float)) for value in corner) for corner in corners):
            raise ValueError("document_localization anchor corners must be numeric")
    return profile
