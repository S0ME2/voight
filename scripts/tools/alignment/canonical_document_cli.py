from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    point_sum = points.sum(axis=1)
    point_diff = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(point_sum)]
    ordered[2] = points[np.argmax(point_sum)]
    ordered[1] = points[np.argmin(point_diff)]
    ordered[3] = points[np.argmax(point_diff)]
    return ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactively select four document corners and warp to a canonical size."
    )
    parser.add_argument("--input", type=Path, required=True, help="Input document image.")
    parser.add_argument("--output", type=Path, required=True, help="Canonical image output path.")
    parser.add_argument("--canonical-width", type=int, default=1000, help="Output width in pixels.")
    parser.add_argument("--canonical-height", type=int, default=630, help="Output height in pixels.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = cv2.imread(str(args.input))
    if image is None:
        raise FileNotFoundError(f"Could not load: {args.input}")

    points: list[tuple[int, int]] = []
    window_name = "Select 4 document corners"

    def callback(event, x, y, flags, param):
        del flags, param
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))
            print(f"Point {len(points)}: ({x}, {y})")

    cv2.namedWindow(window_name)
    cv2.setMouseCallback(window_name, callback)
    while True:
        preview = image.copy()
        for index, point in enumerate(points):
            cv2.circle(preview, point, 6, (0, 255, 0), -1)
            cv2.putText(preview, str(index + 1), (point[0] + 8, point[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow(window_name, preview)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:
            break
        if key == ord("r"):
            points.clear()
        if key in {10, 13} and len(points) == 4:
            break
    cv2.destroyAllWindows()

    if len(points) != 4:
        raise RuntimeError("You must select exactly 4 corners")

    source = order_points(np.array(points, dtype=np.float32))
    destination = np.array(
        [
            [0, 0],
            [args.canonical_width - 1, 0],
            [args.canonical_width - 1, args.canonical_height - 1],
            [0, args.canonical_height - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source, destination)
    canonical = cv2.warpPerspective(
        image, transform, (args.canonical_width, args.canonical_height)
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), canonical):
        raise RuntimeError(f"Could not save: {args.output}")
    print(f"Saved canonical document to: {args.output}")


if __name__ == "__main__":
    main()
