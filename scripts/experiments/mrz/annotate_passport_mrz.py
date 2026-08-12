#!/usr/bin/env python3
"""Diagnostic MRZ overlay helper; use dataset/annotate_profiles.py for profiles."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.documents.passport_localization import relative_to_mrz_width
from app.imaging import draw_polygon, order_corners
from app.models import Models
from app.config import Settings


def relative_to_mrz(mrz: np.ndarray, page: np.ndarray) -> list[list[float]]:
    """Express page corners in the detected MRZ quadrilateral's coordinates."""
    transform = cv2.getPerspectiveTransform(
        order_corners(mrz), np.float32([[0, 0], [1, 0], [1, 1], [0, 1]])
    )
    return cv2.perspectiveTransform(np.asarray(page, dtype=np.float32)[None], transform)[0].tolist()


def click_page_corners(image: np.ndarray) -> np.ndarray:
    points: list[tuple[int, int]] = []
    title = "Click passport page: top-left, top-right, bottom-right, bottom-left (u undo, r reset, Enter save)"

    def click(event, x, y, _flags, _data):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, click)
    while True:
        frame = image.copy()
        for number, point in enumerate(points, 1):
            cv2.circle(frame, point, 5, (0, 0, 255), -1)
            cv2.putText(frame, str(number), point, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow(title, frame)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("u") and points:
            points.pop()
        elif key == ord("r"):
            points.clear()
        elif key in (13, 32) and len(points) == 4:
            cv2.destroyAllWindows()
            return order_corners(np.float32(points))
        elif key in (27, ord("q")):
            cv2.destroyAllWindows()
            raise SystemExit("Cancelled")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--annotate-page", action="store_true", help="open a window to click the four passport-page corners")
    args = parser.parse_args()
    image = cv2.imread(str(args.image))
    if image is None:
        parser.error(f"Cannot read image: {args.image}")
    detected = Models(Settings.from_env()).mrz_scanner()(image, do_center_crop=False)
    polygon = np.asarray(detected["mrz_polygon"], dtype=np.float32).reshape(4, 2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    overlay = draw_polygon(image, polygon)
    overlay_path = args.output_dir / f"{args.image.stem}_mrz.jpg"
    cv2.imwrite(str(overlay_path), overlay)
    record = {"image": str(args.image), "mrz_polygon": order_corners(polygon).tolist()}
    if args.annotate_page:
        page = click_page_corners(overlay)
        record["page_corners"] = page.tolist()
        record["page_corners_relative_to_mrz"] = relative_to_mrz(polygon, page)
        record["page_corners_relative_to_mrz_width"] = relative_to_mrz_width(polygon, page)
        cv2.imwrite(str(args.output_dir / f"{args.image.stem}_mrz_page.jpg"), draw_polygon(overlay, page))
    json_path = args.output_dir / f"{args.image.stem}_mrz.json"
    json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(overlay_path)
    print(json_path)


if __name__ == "__main__":
    main()
