#!/usr/bin/env python3
"""Promote checked source annotations into canonical runtime profiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.documents.passport_localization import relative_to_mrz_width


ROOT = Path(__file__).resolve().parents[1]
FIELD_ALIASES = {"sec": "sex"}


def _rect_points(rect: dict[str, float], width: float, height: float) -> np.ndarray:
    return np.float32([
        [rect["x1"] * width, rect["y1"] * height],
        [rect["x2"] * width, rect["y1"] * height],
        [rect["x2"] * width, rect["y2"] * height],
        [rect["x1"] * width, rect["y2"] * height],
    ])


def _bounds(points: np.ndarray, width: int, height: int) -> dict[str, float]:
    return {
        "x1": float(points[:, 0].min() / width),
        "y1": float(points[:, 1].min() / height),
        "x2": float(points[:, 0].max() / width),
        "y2": float(points[:, 1].max() / height),
    }


def _relative(bounds: dict[str, float], crop: dict[str, float]) -> dict[str, float]:
    crop_width, crop_height = crop["x2"] - crop["x1"], crop["y2"] - crop["y1"]
    return {
        "x1": (bounds["x1"] - crop["x1"]) / crop_width,
        "y1": (bounds["y1"] - crop["y1"]) / crop_height,
        "x2": (bounds["x2"] - crop["x1"]) / crop_width,
        "y2": (bounds["y2"] - crop["y1"]) / crop_height,
    }


def _canonical_size(existing: dict) -> tuple[int, int]:
    size = existing["canonical_size"]
    return int(size["width"]), int(size["height"])


def _canonical_region(sample: dict, width: int, height: int) -> tuple[dict, dict]:
    """Convert legacy source-space annotations or pass through canonical ones."""
    if sample.get("coordinate_space") == "canonical":
        return sample["data_crop"], sample["fields"]
    source_width, source_height = sample["original_size"]["width"], sample["original_size"]["height"]
    source_corners = np.float32(sample["corners"]) * np.float32([source_width, source_height])
    transform = cv2.getPerspectiveTransform(source_corners, np.float32([[0, 0], [width, 0], [width, height], [0, height]]))
    crop = _bounds(cv2.perspectiveTransform(_rect_points(sample["data_crop"], source_width, source_height)[None, :, :], transform)[0], width, height)
    rois = {}
    for name, roi in sample["fields"].items():
        source_roi = {
            "x1": sample["data_crop"]["x1"] + roi["x1"] * (sample["data_crop"]["x2"] - sample["data_crop"]["x1"]),
            "y1": sample["data_crop"]["y1"] + roi["y1"] * (sample["data_crop"]["y2"] - sample["data_crop"]["y1"]),
            "x2": sample["data_crop"]["x1"] + roi["x2"] * (sample["data_crop"]["x2"] - sample["data_crop"]["x1"]),
            "y2": sample["data_crop"]["y1"] + roi["y2"] * (sample["data_crop"]["y2"] - sample["data_crop"]["y1"]),
        }
        bounds = _bounds(cv2.perspectiveTransform(_rect_points(source_roi, source_width, source_height)[None, :, :], transform)[0], width, height)
        rois[name] = _relative(bounds, crop)
    return crop, rois


def promote(annotation_state: Path, config_dir: Path, passport_anchor: Path | None = None) -> None:
    state = json.loads(annotation_state.read_text(encoding="utf-8"))
    grouped: dict[str, list[dict]] = {}
    for sample in state["samples"].values():
        if sample.get("status") == "complete":
            grouped.setdefault(sample["layout"], []).append(sample)

    for layout, samples in grouped.items():
        if layout == "driving_license":
            # The licence input is already canonical; runtime config is generated,
            # never separately annotated.
            sample = samples[0]
            destination = config_dir.parent / "driving_license"
            destination.mkdir(parents=True, exist_ok=True)
            crop, rois = _canonical_region(sample, 1000, 630)
            (destination / "data_crop.json").write_text(json.dumps({"data_crop": crop}, indent=2) + "\n", encoding="utf-8")
            (destination / "field_rois_crop.json").write_text(json.dumps(rois, indent=2) + "\n", encoding="utf-8")
            continue
        destination = config_dir / ("uz_passport" if layout == "uzbekistan_passport" else "uz_id_card") / "profile.json"
        existing = json.loads(destination.read_text(encoding="utf-8"))
        width, height = _canonical_size(existing)
        regions, fields = {}, []
        required = {field["name"]: field["required"] for field in existing["fields"]}
        for sample in samples:
            region = "data_page" if sample["side"] is None else sample["side"]
            crop, raw_rois = _canonical_region(sample, width, height)
            rois = {}
            for name, roi in raw_rois.items():
                name = FIELD_ALIASES.get(name, name)
                rois[name] = roi
                fields.append({"name": name, "region": region, "required": required.get(name, False)})
            regions[region] = {"data_crop": crop, "field_rois": rois}
        profile = {
            **existing,
            "sample_metadata": {**existing["sample_metadata"], "annotation_source": str(annotation_state.relative_to(ROOT))},
            "regions": regions,
            "fields": fields,
        }
        anchor = next((sample.get("mrz_anchor") for sample in samples if sample.get("mrz_anchor")), None)
        if layout == "uzbekistan_passport" and (anchor or passport_anchor is not None):
            anchor = anchor or json.loads(passport_anchor.read_text(encoding="utf-8"))
            profile["document_localization"] = {
                "strategy": "mrz_anchor",
                "page_corners_relative_to_mrz_width": anchor.get("page_corners_relative_to_mrz_width")
                or relative_to_mrz_width(anchor["mrz_polygon"], anchor["page_corners"]),
            }
        else:
            profile.pop("document_localization", None)
        destination.write_text(json.dumps(profile, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, default=ROOT / "annotations" / "annotation_state.json")
    parser.add_argument("--config-dir", type=Path, default=ROOT / "config" / "documents")
    parser.add_argument("--passport-mrz-anchor", type=Path, default=ROOT / "annotations" / "previews" / "passport_mrz.json")
    args = parser.parse_args()
    promote(args.annotations, args.config_dir, args.passport_mrz_anchor if args.passport_mrz_anchor.is_file() else None)


if __name__ == "__main__":
    main()
