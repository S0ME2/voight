from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively select the useful-data crop on a canonical document image."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("assets/samples/driving_license/test_license_canonical.jpg"),
        help="Canonical document image used to select the crop.",
    )
    parser.add_argument(
        "--output-config",
        type=Path,
        default=Path("config/driving_license/data_crop.json"),
        help="JSON file that receives normalized crop coordinates.",
    )
    parser.add_argument(
        "--output-preview",
        type=Path,
        default=Path("outputs/tools/roi/data_crop_preview.jpg"),
        help="Annotated preview image output.",
    )
    parser.add_argument(
        "--output-crop",
        type=Path,
        default=Path("outputs/tools/roi/data_crop.jpg"),
        help="Selected cropped image output.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.input))
    if image is None:
        raise FileNotFoundError(f"Could not load image: {args.input}")

    image_height, image_width = image.shape[:2]
    x, y, width, height = cv2.selectROI(
        "Select useful data region", image, showCrosshair=True, fromCenter=False
    )
    cv2.destroyAllWindows()
    if width == 0 or height == 0:
        raise RuntimeError("No ROI was selected")

    x1, y1, x2, y2 = int(x), int(y), int(x + width), int(y + height)
    data = {
        "data_crop": {
            "x1": x1 / image_width,
            "y1": y1 / image_height,
            "x2": x2 / image_width,
            "y2": y2 / image_height,
        }
    }

    args.output_config.parent.mkdir(parents=True, exist_ok=True)
    args.output_preview.parent.mkdir(parents=True, exist_ok=True)
    args.output_crop.parent.mkdir(parents=True, exist_ok=True)
    args.output_config.write_text(json.dumps(data, indent=4), encoding="utf-8")

    crop = image[y1:y2, x1:x2]
    if not cv2.imwrite(str(args.output_crop), crop):
        raise RuntimeError(f"Could not save crop: {args.output_crop}")

    preview = image.copy()
    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 0), 3)
    if not cv2.imwrite(str(args.output_preview), preview):
        raise RuntimeError(f"Could not save preview: {args.output_preview}")

    print(f"Crop configuration: {args.output_config}")
    print(f"Crop preview:       {args.output_preview}")
    print(f"Cropped image:      {args.output_crop}")


if __name__ == "__main__":
    main()
