import re
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from app.artifacts import ArtifactWriter
from app.config import MrzSettings
from app.contracts import MrzResult, ValidationResult, ValidationStatus
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


def _mrz_value(character: str) -> int:
    if character.isdigit():
        return int(character)
    if "A" <= character <= "Z":
        return ord(character) - ord("A") + 10
    if character == "<":
        return 0
    raise ValueError(f"invalid MRZ character: {character!r}")


def check_digit(value: str) -> str:
    """Return the ICAO 9303 check digit for one MRZ field."""
    weights = (7, 3, 1)
    return str(sum(_mrz_value(character) * weights[index % 3] for index, character in enumerate(value)) % 10)


def _name_fields(value: str) -> tuple[str | None, str | None]:
    surname, separator, names = value.partition("<<")
    clean = lambda text: " ".join(text.replace("<", " ").split()) or None
    return clean(surname), clean(names if separator else "")


def _check(code: str, value: str, expected: str, fields: list[str]) -> ValidationResult:
    actual = check_digit(value)
    return ValidationResult(
        code=code,
        status=ValidationStatus.PASSED if expected == actual else ValidationStatus.FAILED,
        detail=f"expected {expected}, calculated {actual}",
        fields=fields,
    )


def _lines(text: str, document_type: str) -> list[str]:
    lines = [_fragment(line) for line in text.splitlines() if _fragment(line)]
    if len(lines) == 1:
        widths = {"passport": (44, 2), "id_card": (30, 3)}
        width, count = widths[document_type]
        if len(lines[0]) == width * count:
            lines = [lines[0][offset : offset + width] for offset in range(0, width * count, width)]
    return lines


def parse(text: str, document_type: str) -> MrzResult:
    """Parse the supplied Uzbekistan passport TD3 or ID-card TD1 MRZ."""
    if document_type not in {"passport", "id_card"}:
        raise ValueError(f"unsupported MRZ document type: {document_type}")
    lines = _lines(text, document_type)
    expected_shape = (2, 44) if document_type == "passport" else (3, 30)
    if not lines:
        return MrzResult(
            validations=[
                ValidationResult(
                    code="mrz_present",
                    status=ValidationStatus.NOT_RUN,
                    detail="MRZ was not found",
                )
            ]
        )
    if len(lines) != expected_shape[0] or any(len(line) != expected_shape[1] for line in lines):
        return MrzResult(
            raw_lines=lines,
            validations=[
                ValidationResult(
                    code="mrz_format",
                    status=ValidationStatus.FAILED,
                    detail=f"expected {expected_shape[0]} lines of {expected_shape[1]} characters",
                )
            ],
        )

    if document_type == "passport":
        first, second = lines
        surname, given_names = _name_fields(first[5:44])
        fields = {
            "document_code": first[0:2].replace("<", "") or None,
            "issuing_state": first[2:5].replace("<", "") or None,
            "surname": surname,
            "given_names": given_names,
            "document_number": second[0:9].replace("<", "") or None,
            "nationality": second[10:13].replace("<", "") or None,
            "date_of_birth": second[13:19].replace("<", "") or None,
            "sex": second[20].replace("<", "") or None,
            "date_of_expiry": second[21:27].replace("<", "") or None,
            "optional_data": second[28:42].replace("<", "") or None,
        }
        validations = [
            _check("mrz_document_number_check_digit", second[0:9], second[9], ["document_number"]),
            _check("mrz_birth_date_check_digit", second[13:19], second[19], ["date_of_birth"]),
            _check("mrz_expiry_date_check_digit", second[21:27], second[27], ["date_of_expiry"]),
            _check("mrz_optional_data_check_digit", second[28:42], second[42], ["optional_data"]),
            _check(
                "mrz_composite_check_digit",
                second[0:10] + second[13:20] + second[21:43],
                second[43],
                ["document_number", "date_of_birth", "date_of_expiry", "optional_data"],
            ),
        ]
    else:
        first, second, third = lines
        surname, given_names = _name_fields(third)
        fields = {
            "document_code": first[0:2].replace("<", "") or None,
            "issuing_state": first[2:5].replace("<", "") or None,
            "document_number": first[5:14].replace("<", "") or None,
            "optional_data": first[15:30].replace("<", "") or None,
            "pinfl": first[15:30].replace("<", "") or None,
            "date_of_birth": second[0:6].replace("<", "") or None,
            "sex": second[7].replace("<", "") or None,
            "date_of_expiry": second[8:14].replace("<", "") or None,
            "nationality": second[15:18].replace("<", "") or None,
            "surname": surname,
            "given_names": given_names,
        }
        validations = [
            _check("mrz_document_number_check_digit", first[5:14], first[14], ["document_number"]),
            _check("mrz_birth_date_check_digit", second[0:6], second[6], ["date_of_birth"]),
            _check("mrz_expiry_date_check_digit", second[8:14], second[14], ["date_of_expiry"]),
            _check(
                "mrz_composite_check_digit",
                first[5:30] + second[0:7] + second[8:15] + second[18:29],
                second[29],
                ["document_number", "date_of_birth", "date_of_expiry", "optional_data"],
            ),
        ]
    return MrzResult(raw_lines=lines, fields=fields, validations=validations)


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
