from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


DEFAULT_FIELDS = [
    "surname",
    "given_names",
    "birth_place_and_date",
    "issue_date",
    "expiry_date",
    "issued_place",
    "personal_id",
    "license_number",
    "address",
    "categories",
    "serial_number",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively create normalized field ROIs for a cropped document data region."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("assets/samples/driving_license/test_license_data_crop.jpg"),
        help="Cropped data-region image used to define field rectangles.",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("config/driving_license/field_rois_crop.json"),
        help="JSON file that receives normalized field ROI coordinates.",
    )
    parser.add_argument(
        "--output-preview",
        type=Path,
        default=Path("outputs/tools/roi/field_rois_crop_preview.jpg"),
        help="Annotated preview image output.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        default=DEFAULT_FIELDS,
        help="Field names to select, in selection order.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.input))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {args.input}")

    image_height, image_width = image.shape[:2]
    preview = image.copy()
    rois: dict[str, dict[str, float]] = {}

    for field_name in args.fields:
        x, y, width, height = cv2.selectROI(
            f"Select ROI: {field_name}", preview, showCrosshair=True, fromCenter=False
        )
        cv2.destroyWindow(f"Select ROI: {field_name}")
        if width == 0 or height == 0:
            print(f"Skipped: {field_name}")
            continue

        x1, y1, x2, y2 = int(x), int(y), int(x + width), int(y + height)
        rois[field_name] = {
            "x1": x1 / image_width,
            "y1": y1 / image_height,
            "x2": x2 / image_width,
            "y2": y2 / image_height,
        }
        cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            preview,
            field_name,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_preview.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(
        json.dumps(rois, ensure_ascii=False, indent=4), encoding="utf-8"
    )
    if not cv2.imwrite(str(args.output_preview), preview):
        raise RuntimeError(f"Could not save preview: {args.output_preview}")

    print(f"ROIs saved to: {args.output_config}")
    print(f"Preview saved to: {args.output_preview}")
    print(f"Created {len(rois)} of {len(args.fields)} ROIs")


if __name__ == "__main__":
    main()
