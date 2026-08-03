import cv2
import numpy as np


def order_corners(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(4, 2)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0], ordered[2] = points[np.argmin(sums)], points[np.argmax(sums)]
    ordered[1], ordered[3] = points[np.argmin(differences)], points[np.argmax(differences)]
    return ordered


def warp_to_size(image: np.ndarray, corners: np.ndarray, width: int, height: int) -> np.ndarray:
    source = order_corners(corners)
    destination = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], dtype=np.float32)
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(source, destination), (width, height))


def draw_polygon(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    output = image.copy()
    cv2.polylines(output, [order_corners(corners).astype(np.int32)], True, (0, 255, 0), 4)
    return output
