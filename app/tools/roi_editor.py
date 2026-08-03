import argparse
import json
from pathlib import Path

import cv2

from app.roi import crop_normalized_roi, draw_roi_assignments, load_roi_config


def _image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(path)
    return image


def _select(image, title: str) -> dict[str, float] | None:
    x, y, width, height = cv2.selectROI(title, image, showCrosshair=True, fromCenter=False)
    if not width or not height:
        return None
    image_height, image_width = image.shape[:2]
    return {"x1": x / image_width, "y1": y / image_height, "x2": (x + width) / image_width, "y2": (y + height) / image_height}


def select_crop(args: argparse.Namespace) -> None:
    image = _image(args.image)
    roi = _select(image, "Select data crop")
    cv2.destroyAllWindows()
    if roi is None:
        raise RuntimeError("No crop selected")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"data_crop": roi}, indent=2), encoding="utf-8")
    cv2.imwrite(str(args.preview), crop_normalized_roi(image, roi))


def select_fields(args: argparse.Namespace) -> None:
    image = _image(args.image)
    rois = {}
    for name in args.fields:
        roi = _select(image, f"Select ROI: {name}")
        if roi is not None:
            rois[name] = roi
    cv2.destroyAllWindows()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.preview.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rois, ensure_ascii=False, indent=2), encoding="utf-8")
    cv2.imwrite(str(args.preview), draw_roi_assignments(image, rois, {}))


def preview(args: argparse.Namespace) -> None:
    image = _image(args.image)
    rois = load_roi_config(args.rois)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(args.output), draw_roi_assignments(image, rois, {}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create or preview normalized ROI configuration.")
    commands = parser.add_subparsers(required=True)
    for name, handler in (("crop", select_crop), ("fields", select_fields)):
        command = commands.add_parser(name)
        command.add_argument("image", type=Path)
        command.add_argument("output", type=Path)
        command.add_argument("--preview", type=Path, default=Path("roi_preview.jpg"))
        if name == "fields":
            command.add_argument("fields", nargs="+")
        command.set_defaults(handler=handler)
    command = commands.add_parser("preview")
    command.add_argument("image", type=Path)
    command.add_argument("rois", type=Path)
    command.add_argument("output", type=Path)
    command.set_defaults(handler=preview)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
