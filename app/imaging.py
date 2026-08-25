import cv2
import numpy as np


PREPROCESSING_VARIANTS = {
    "original",
    "grayscale",
    "contrast_1.15",
    "contrast_1.25",
    "contrast_1.30",
    "contrast_1.50",
    "clahe_mild",
    "clahe_medium",
    "gamma_0.8",
    "gamma_1.2",
    "sharpen_light",
    "otsu",
    "adaptive",
}


def preprocess_variant(image: np.ndarray, variant: str) -> np.ndarray:
    """Apply one small, benchmark-only image transform in BGR channel shape."""
    if variant == "original":
        return image.copy()
    if variant not in PREPROCESSING_VARIANTS:
        raise ValueError(f"unknown preprocessing variant: {variant}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    if variant.startswith("contrast_"):
        factor = float(variant.removeprefix("contrast_"))
        mean = float(gray.mean())
        gray = np.clip((gray.astype(np.float32) - mean) * factor + mean, 0, 255).astype(np.uint8)
    elif variant.startswith("clahe_"):
        clip = 1.5 if variant == "clahe_mild" else 2.5
        gray = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(gray)
    elif variant.startswith("gamma_"):
        gamma = float(variant.removeprefix("gamma_"))
        table = np.array([((value / 255.0) ** gamma) * 255 for value in range(256)], dtype=np.uint8)
        gray = cv2.LUT(gray, table)
    elif variant == "sharpen_light":
        gray = cv2.addWeighted(gray, 1.25, cv2.GaussianBlur(gray, (3, 3), 0), -0.25, 0)
    elif variant == "otsu":
        _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif variant == "adaptive":
        gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 11)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


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
