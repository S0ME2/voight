import re
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.artifacts import ArtifactWriter
from app.config import MrzSettings
from app.imaging import draw_polygon, order_corners
from app.ocr import recognize

MRZ_ALLOWED_RE = re.compile(r"^[A-Z0-9<]+$")
VALID_MRZ_LENGTHS = (30, 36, 44)


@dataclass(frozen=True)
class MRZLine:
    text: str
    confidence: float
    center_y: float


def preprocess(image: np.ndarray, max_side: int, contrast: float) -> np.ndarray:
    height, width = image.shape[:2]
    if max(height, width) > max_side:
        scale = max_side / max(height, width)
        image = cv2.resize(image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    adjusted = np.clip((gray.astype(np.float32) - mean) * contrast + mean, 0, 255).astype(np.uint8)
    return cv2.cvtColor(adjusted, cv2.COLOR_GRAY2BGR)


def crop_polygon(image: np.ndarray, polygon: np.ndarray, padding: float) -> tuple[np.ndarray, np.ndarray]:
    polygon = np.asarray(polygon, dtype=np.float32).reshape(4, 2)
    height, width = image.shape[:2]
    center = polygon.mean(axis=0)
    expanded = center + (polygon - center) * (1 + padding)
    expanded[:, 0] = np.clip(expanded[:, 0], 0, width - 1)
    expanded[:, 1] = np.clip(expanded[:, 1], 0, height - 1)
    tl, tr, br, bl = order_corners(expanded)
    crop_width = max(1, round(max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))))
    crop_height = max(1, round(max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))))
    target = np.array([[0, 0], [crop_width - 1, 0], [crop_width - 1, crop_height - 1], [0, crop_height - 1]], dtype=np.float32)
    return cv2.warpPerspective(image, cv2.getPerspectiveTransform(np.array([tl, tr, br, bl]), target), (crop_width, crop_height)), expanded


def _fragment(text: str) -> str:
    return re.sub(r"[^A-Z0-9<]", "", text.upper())


def reconstruct(tokens: list[dict[str, Any]]) -> list[MRZLine]:
    tokens = [token for token in tokens if _fragment(token["text"])]
    if not tokens:
        return []
    threshold = max(5.0, float(np.median([token["height"] for token in tokens])) * 0.7)
    clusters: list[list[dict[str, Any]]] = []
    for token in sorted(tokens, key=lambda item: (item["center_y"], item["x1"])):
        cluster = min(clusters, key=lambda items: abs(token["center_y"] - np.mean([item["center_y"] for item in items])), default=None)
        if cluster is not None and abs(token["center_y"] - np.mean([item["center_y"] for item in cluster])) <= threshold:
            cluster.append(token)
        else:
            clusters.append([token])
    lines = []
    for cluster in clusters:
        ordered = sorted(cluster, key=lambda item: item["x1"])
        text = "".join(_fragment(item["text"]) for item in ordered)
        if len(text) >= 20 and MRZ_ALLOWED_RE.fullmatch(text):
            weight = sum(max(1, len(_fragment(item["text"]))) for item in ordered)
            confidence = sum(
                item["score"] * max(1, len(_fragment(item["text"]))) for item in ordered
            ) / weight
            lines.append(MRZLine(text, confidence, float(np.mean([item["center_y"] for item in ordered]))))
    return lines


def select(lines: list[MRZLine], counts: tuple[int, ...]) -> list[MRZLine]:
    candidates = [line for line in lines if 24 <= len(line.text) <= 50]
    best, best_score = [], float("-inf")
    for count in counts:
        for start in range(max(0, len(candidates) - count + 1)):
            block = candidates[start : start + count]
            lengths = [len(line.text) for line in block]
            score = sum(max(0.0, 1 - min(abs(length - target) for target in VALID_MRZ_LENGTHS) / 12) for length in lengths)
            score += sum(0.4 for line in block if "<" in line.text) + 0.75 * (1 - min(1, (max(lengths) - min(lengths)) / 12))
            score += 0.15 * sum(line.center_y for line in block) / max(1.0, max((line.center_y for line in candidates), default=1.0))
            if score > best_score:
                best, best_score = block, score
    return best


@dataclass(frozen=True)
class MrzProfile:
    line_counts: tuple[int, ...]


ID_CARD = MrzProfile((3, 2))
PASSPORT = MrzProfile((2,))


def extract(
    image: np.ndarray,
    detector: Any,
    ocr: Any,
    artifacts: ArtifactWriter,
    settings: MrzSettings,
    profile: MrzProfile,
    *,
    started_total: float | None = None,
    initial_timings: dict[str, Any] | None = None,
) -> tuple[str, dict[str, Any]]:
    timings: dict[str, Any] = dict(initial_timings or {})
    total = started_total if started_total is not None else time.perf_counter()
    started = time.perf_counter()
    detected = detector(image, do_center_crop=False)
    timings["mrz_detection_seconds"] = time.perf_counter() - started
    polygon = np.asarray(detected["mrz_polygon"], dtype=np.float32).reshape(4, 2)
    artifacts.save_json("01_mrzscanner_result.json", detected)
    artifacts.save_image("02_mrz_detection.jpg", draw_polygon(image, polygon))
    started = time.perf_counter()
    crop, expanded = crop_polygon(image, polygon, settings.polygon_padding_ratio)
    timings["perspective_crop_seconds"] = time.perf_counter() - started
    artifacts.save_json("03_mrz_polygons.json", {"detected_polygon": polygon, "expanded_polygon": expanded, "padding_ratio": settings.polygon_padding_ratio})
    artifacts.save_image("04_mrz_crop.jpg", crop)
    processed = preprocess(crop, settings.max_side, settings.contrast)
    artifacts.save_image("05_preprocessed_mrz.png", processed)
    started = time.perf_counter()
    tokens = recognize(processed, ocr, artifacts)
    timings["ocr_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    lines = reconstruct(tokens)
    selected = select(lines, profile.line_counts)
    text = "\n".join(line.text for line in selected)
    timings["reconstruction_seconds"] = time.perf_counter() - started
    timings["total_seconds"] = time.perf_counter() - total
    artifacts.save_json("08_ocr_tokens.json", tokens)
    artifacts.save_json("09_reconstructed_mrz.json", {"all_lines": [line.text for line in lines], "selected_lines": [line.text for line in selected]})
    artifacts.save_text("10_mrz.txt", text)
    artifacts.save_json("11_timings.json", timings)
    return text, timings
