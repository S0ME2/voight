from __future__ import annotations

import argparse
import csv
import gc
import itertools
import json
import math
import random
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import cv2
import numpy as np
from paddleocr import PaddleOCR


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
WHITESPACE_RE = re.compile(r"\s+")


# ============================================================================
# MODEL SCREEN
# ============================================================================


@dataclass(frozen=True)
class ModelSpec:
    name: str
    detection_model: str
    recognition_model: str


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "v6_medium",
        "PP-OCRv6_medium_det",
        "PP-OCRv6_medium_rec",
    ),
    ModelSpec(
        "v6_small",
        "PP-OCRv6_small_det",
        "PP-OCRv6_small_rec",
    ),
    ModelSpec(
        "v6_tiny",
        "PP-OCRv6_tiny_det",
        "PP-OCRv6_tiny_rec",
    ),
    ModelSpec(
        "v5_server",
        "PP-OCRv5_server_det",
        "PP-OCRv5_server_rec",
    ),
    ModelSpec(
        "v5_mobile",
        "PP-OCRv5_mobile_det",
        "PP-OCRv5_mobile_rec",
    ),
    ModelSpec(
        "v5_server_det_en_rec",
        "PP-OCRv5_server_det",
        "en_PP-OCRv5_mobile_rec",
    ),
)


PREPROCESSOR_NAMES = (
    "original",
    "gray",
    "clahe",
    "upscale_1_5",
    "clahe_upscale_1_5",
)


PIPELINE_FLAG_SETS = (
    (False, False, False),
    (True, False, False),
    (False, False, True),
    (True, False, True),
)


DETECTION_PARAM_SETS = (
    (0.30, 0.60, 2.00),
    (0.20, 0.60, 2.00),
    (0.30, 0.50, 2.00),
)


@dataclass(frozen=True)
class Sample:
    image_path: Path
    image_key: str
    label_id: str
    expected_mrz: tuple[str, ...]
    variant: str
    augmentation_types: tuple[str, ...]


@dataclass(frozen=True)
class Config:
    stage: str
    model_name: str
    detection_model: str
    recognition_model: str
    preprocessing: str = "original"
    use_doc_orientation_classify: bool = False
    use_doc_unwarping: bool = False
    use_textline_orientation: bool = False
    text_det_thresh: float = 0.30
    text_det_box_thresh: float = 0.60
    text_det_unclip_ratio: float = 2.00

    @property
    def ocr_key(self) -> tuple[Any, ...]:
        return (
            self.detection_model,
            self.recognition_model,
            self.use_doc_orientation_classify,
            self.use_doc_unwarping,
            self.use_textline_orientation,
        )

    @property
    def config_id(self) -> str:
        return (
            f"{self.stage}__{self.model_name}__{self.preprocessing}"
            f"__ori{int(self.use_doc_orientation_classify)}"
            f"_uw{int(self.use_doc_unwarping)}"
            f"_tl{int(self.use_textline_orientation)}"
            f"__dt{self.text_det_thresh:.2f}"
            f"_db{self.text_det_box_thresh:.2f}"
            f"_du{self.text_det_unclip_ratio:.2f}"
        )


@dataclass
class OCRLine:
    text: str
    score: float
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(1.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(1.0, self.y2 - self.y1)

    @property
    def center_y(self) -> float:
        return (self.y1 + self.y2) / 2.0


@dataclass
class Prediction:
    config_id: str
    stage: str
    image_key: str
    label_id: str
    variant: str
    augmentation_types: str
    expected_mrz: str
    predicted_mrz: str
    mrz_found: bool
    full_exact_match: bool
    line_exact_rate: float
    char_accuracy: float
    edit_distance: int
    mean_mrz_confidence: float | None
    check_digits_valid: bool | None
    latency_seconds: float
    error: str | None


# ============================================================================
# DATASET
# ============================================================================


def normalize_mrz_line(text: str) -> str:
    return WHITESPACE_RE.sub("", text.upper().strip())


def load_labels(path: Path) -> dict[str, tuple[str, ...]]:
    with path.open("r", encoding="utf-8") as file:
        raw = json.load(file)

    labels: dict[str, tuple[str, ...]] = {}

    for key, value in raw.items():
        mrz = value.get("mrz") if isinstance(value, dict) else value

        if not isinstance(mrz, list) or not mrz:
            raise ValueError(f"{key!r}: expected a non-empty MRZ list")

        labels[str(key)] = tuple(normalize_mrz_line(str(line)) for line in mrz)

    return labels


def augmentation_types(record: dict[str, Any]) -> tuple[str, ...]:
    augmentation = record.get("augmentation")
    if not augmentation:
        return ("original",)

    operations = augmentation.get("operations", [])
    values = tuple(
        str(item["type"])
        for item in operations
        if isinstance(item, dict) and item.get("type")
    )

    return values or ("augmented",)


def load_samples(
    dataset_root: Path,
    labels_path: Path | None,
    manifest_path: Path | None,
) -> list[Sample]:
    labels_path = labels_path or dataset_root / "labels.json"
    manifest_path = manifest_path or dataset_root / "generated" / "manifest.jsonl"

    labels = load_labels(labels_path)

    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Manifest not found: {manifest_path}\n"
            "This benchmark expects the augmented dataset manifest."
        )

    samples: list[Sample] = []
    base = manifest_path.parent

    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            record = json.loads(line)
            label_id = str(record.get("label_id") or record.get("source_image") or "")

            if label_id not in labels:
                raise KeyError(
                    f"{manifest_path}:{line_number}: "
                    f"label_id={label_id!r} not found in labels.json"
                )

            relative = Path(str(record["image"]))
            image_path = relative if relative.is_absolute() else base / relative

            if not image_path.exists():
                raise FileNotFoundError(f"Missing image: {image_path}")

            samples.append(
                Sample(
                    image_path=image_path,
                    image_key=str(relative),
                    label_id=label_id,
                    expected_mrz=labels[label_id],
                    variant=str(record.get("variant", "unknown")),
                    augmentation_types=augmentation_types(record),
                )
            )

    if not samples:
        raise RuntimeError("No samples found.")

    return samples


def diverse_sample(
    samples: Sequence[Sample],
    count: int,
    seed: int,
) -> list[Sample]:
    if count >= len(samples):
        return list(samples)

    rng = random.Random(seed)
    groups: dict[str, list[Sample]] = {}

    for sample in samples:
        groups.setdefault(sample.label_id, []).append(sample)

    for group in groups.values():
        rng.shuffle(group)

    label_ids = list(groups)
    rng.shuffle(label_ids)

    chosen: list[Sample] = []
    depth = 0

    while len(chosen) < count:
        added = False

        for label_id in label_ids:
            group = groups[label_id]

            if depth < len(group):
                chosen.append(group[depth])
                added = True

                if len(chosen) == count:
                    return chosen

        if not added:
            break

        depth += 1

    return chosen


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================


def limit_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    current = max(height, width)

    if current <= max_side:
        return image

    scale = max_side / current

    return cv2.resize(
        image,
        (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )


def original(image: np.ndarray) -> np.ndarray:
    return image


def gray(image: np.ndarray) -> np.ndarray:
    value = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)


def clahe(image: np.ndarray) -> np.ndarray:
    value = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    value = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8),
    ).apply(value)
    return cv2.cvtColor(value, cv2.COLOR_GRAY2BGR)


def upscale_1_5(image: np.ndarray) -> np.ndarray:
    return cv2.resize(
        image,
        None,
        fx=1.5,
        fy=1.5,
        interpolation=cv2.INTER_CUBIC,
    )


def clahe_upscale_1_5(image: np.ndarray) -> np.ndarray:
    return upscale_1_5(clahe(image))


PREPROCESSORS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "original": original,
    "gray": gray,
    "clahe": clahe,
    "upscale_1_5": upscale_1_5,
    "clahe_upscale_1_5": clahe_upscale_1_5,
}


# ============================================================================
# PADDLEOCR OUTPUT PARSING
# ============================================================================


def result_field(result: Any, key: str, default: Any = None) -> Any:
    try:
        return result[key]
    except (KeyError, TypeError, AttributeError):
        pass

    try:
        data = result.json
        if callable(data):
            data = data()

        if isinstance(data, dict):
            if key in data:
                return data[key]

            if isinstance(data.get("res"), dict):
                return data["res"].get(key, default)
    except Exception:
        pass

    return default


def to_list(value: Any) -> list[Any]:
    if value is None:
        return []

    if isinstance(value, np.ndarray):
        return value.tolist()

    return list(value)


def extract_ocr_lines(results: Iterable[Any]) -> list[OCRLine]:
    lines: list[OCRLine] = []

    for result in results:
        texts = to_list(result_field(result, "rec_texts", None))
        scores = to_list(result_field(result, "rec_scores", None))
        boxes = to_list(result_field(result, "rec_boxes", None))

        if len(boxes) != len(texts):
            boxes = [
                [0, index * 20, max(10, len(str(text)) * 12), index * 20 + 16]
                for index, text in enumerate(texts)
            ]

        for index, text in enumerate(texts):
            cleaned = normalize_mrz_line(str(text))

            if not cleaned:
                continue

            score = float(scores[index]) if index < len(scores) else 0.0
            flat = np.asarray(boxes[index]).reshape(-1)

            if flat.size >= 4:
                x1, y1, x2, y2 = map(float, flat[:4])
            else:
                x1, y1, x2, y2 = 0.0, index * 20.0, 100.0, index * 20.0 + 16.0

            lines.append(
                OCRLine(
                    text=cleaned,
                    score=score,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

    return lines


def mrz_charset_ratio(text: str) -> float:
    if not text:
        return 0.0

    allowed = sum(
        char == "<" or char.isdigit() or ("A" <= char <= "Z") for char in text
    )

    return allowed / len(text)


def merge_same_row(lines: Sequence[OCRLine]) -> list[OCRLine]:
    if not lines:
        return []

    clusters: list[list[OCRLine]] = []

    for line in sorted(lines, key=lambda item: item.center_y):
        match: list[OCRLine] | None = None

        for cluster in clusters:
            center = statistics.mean(item.center_y for item in cluster)
            height = statistics.median(item.height for item in cluster)

            if abs(line.center_y - center) <= max(height, line.height) * 0.65:
                match = cluster
                break

        if match is None:
            clusters.append([line])
        else:
            match.append(line)

    merged: list[OCRLine] = []

    for cluster in clusters:
        ordered = sorted(cluster, key=lambda item: item.x1)
        text = "".join(item.text for item in ordered)

        weight = sum(max(1, len(item.text)) for item in ordered)
        score = sum(item.score * max(1, len(item.text)) for item in ordered) / weight

        merged.append(
            OCRLine(
                text=text,
                score=score,
                x1=min(item.x1 for item in ordered),
                y1=min(item.y1 for item in ordered),
                x2=max(item.x2 for item in ordered),
                y2=max(item.y2 for item in ordered),
            )
        )

    return merged


def candidate_score(
    line: OCRLine,
    expected_length: int,
    image_width: int,
    image_height: int,
) -> float:
    length_score = max(
        0.0,
        1.0 - abs(len(line.text) - expected_length) / max(1, expected_length),
    )

    return (
        5.0 * length_score
        + 3.0 * mrz_charset_ratio(line.text)
        + (0.5 if "<" in line.text else 0.0)
        + 0.5 * min(1.0, line.width / max(1.0, image_width * 0.5))
        + 0.25 * min(1.0, line.center_y / max(1.0, image_height))
    )


def select_mrz(
    lines: Sequence[OCRLine],
    expected: Sequence[str],
    image_shape: tuple[int, ...],
) -> tuple[tuple[str, ...] | None, float | None]:
    if not lines:
        return None, None

    height, width = image_shape[:2]
    target_lengths = tuple(len(line) for line in expected)
    target_count = len(target_lengths)

    candidates = [
        line
        for line in list(lines) + merge_same_row(lines)
        if len(line.text) >= 15 and mrz_charset_ratio(line.text) >= 0.65
    ]

    if len(candidates) < target_count:
        return None, None

    median_target = int(statistics.median(target_lengths))

    candidates = sorted(
        candidates,
        key=lambda item: candidate_score(
            item,
            median_target,
            width,
            height,
        ),
        reverse=True,
    )[:12]

    best: tuple[OCRLine, ...] | None = None
    best_score = -math.inf

    for combo in itertools.combinations(candidates, target_count):
        ordered = tuple(sorted(combo, key=lambda item: item.center_y))

        if any(
            abs(ordered[i + 1].center_y - ordered[i].center_y)
            < 0.25 * max(ordered[i].height, ordered[i + 1].height)
            for i in range(len(ordered) - 1)
        ):
            continue

        score = sum(
            candidate_score(
                line,
                target_lengths[index],
                width,
                height,
            )
            for index, line in enumerate(ordered)
        )

        if score > best_score:
            best_score = score
            best = ordered

    if best is None:
        return None, None

    return (
        tuple(line.text for line in best),
        statistics.mean(line.score for line in best),
    )


# ============================================================================
# METRICS
# ============================================================================


def levenshtein(left: str, right: str) -> int:
    if left == right:
        return 0

    previous = list(range(len(right) + 1))

    for i, left_char in enumerate(left, start=1):
        current = [i]

        for j, right_char in enumerate(right, start=1):
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (left_char != right_char),
                )
            )

        previous = current

    return previous[-1]


def text_metrics(
    expected: Sequence[str],
    predicted: Sequence[str] | None,
) -> tuple[bool, float, int, float]:
    expected_tuple = tuple(expected)
    predicted_tuple = tuple(predicted or ())

    exact = expected_tuple == predicted_tuple

    line_exact = sum(
        index < len(predicted_tuple) and predicted_tuple[index] == expected_line
        for index, expected_line in enumerate(expected_tuple)
    ) / max(1, len(expected_tuple))

    expected_text = "\n".join(expected_tuple)
    predicted_text = "\n".join(predicted_tuple)

    distance = levenshtein(expected_text, predicted_text)
    denominator = max(1, len(expected_text), len(predicted_text))
    char_accuracy = max(0.0, 1.0 - distance / denominator)

    return exact, line_exact, distance, char_accuracy


MRZ_WEIGHTS = (7, 3, 1)


def mrz_value(char: str) -> int:
    if char == "<":
        return 0

    if char.isdigit():
        return int(char)

    if "A" <= char <= "Z":
        return ord(char) - ord("A") + 10

    raise ValueError(char)


def check_digit(data: str) -> str:
    total = sum(
        mrz_value(char) * MRZ_WEIGHTS[index % 3] for index, char in enumerate(data)
    )

    return str(total % 10)


def matches_check_digit(data: str, digit: str) -> bool:
    return digit.isdigit() and check_digit(data) == digit


def validate_check_digits(lines: Sequence[str]) -> bool | None:
    lines = tuple(lines)

    if len(lines) == 2 and all(len(line) == 44 for line in lines):
        line = lines[1]

        checks = [
            matches_check_digit(line[0:9], line[9]),
            matches_check_digit(line[13:19], line[19]),
            matches_check_digit(line[21:27], line[27]),
        ]

        if line[42] != "<":
            checks.append(matches_check_digit(line[28:42], line[42]))

        composite = line[0:10] + line[13:20] + line[21:43]
        checks.append(matches_check_digit(composite, line[43]))

        return all(checks)

    if len(lines) == 2 and all(len(line) == 36 for line in lines):
        line = lines[1]

        return all(
            (
                matches_check_digit(line[0:9], line[9]),
                matches_check_digit(line[13:19], line[19]),
                matches_check_digit(line[21:27], line[27]),
                matches_check_digit(
                    line[0:10] + line[13:20] + line[21:35],
                    line[35],
                ),
            )
        )

    if len(lines) == 3 and all(len(line) == 30 for line in lines):
        first, second, _third = lines

        return all(
            (
                matches_check_digit(first[5:14], first[14]),
                matches_check_digit(second[0:6], second[6]),
                matches_check_digit(second[8:14], second[14]),
                matches_check_digit(
                    first[5:30] + second[0:7] + second[8:15] + second[18:29],
                    second[29],
                ),
            )
        )

    return None


# ============================================================================
# OCR EXECUTION
# ============================================================================


def build_ocr(config: Config, device: str) -> PaddleOCR:
    return PaddleOCR(
        text_detection_model_name=config.detection_model,
        text_recognition_model_name=config.recognition_model,
        use_doc_orientation_classify=config.use_doc_orientation_classify,
        use_doc_unwarping=config.use_doc_unwarping,
        use_textline_orientation=config.use_textline_orientation,
        device=device,
    )


def predict(
    ocr: PaddleOCR,
    image: np.ndarray,
    config: Config,
) -> list[Any]:
    return list(
        ocr.predict(
            image,
            text_det_thresh=config.text_det_thresh,
            text_det_box_thresh=config.text_det_box_thresh,
            text_det_unclip_ratio=config.text_det_unclip_ratio,
            text_rec_score_thresh=0.0,
        )
    )


def evaluate_config(
    ocr: PaddleOCR,
    config: Config,
    samples: Sequence[Sample],
    max_side: int,
    print_errors: int = 2,
) -> tuple[dict[str, Any], list[Prediction]]:
    predictions: list[Prediction] = []
    printed_errors = 0

    for index, sample in enumerate(samples, start=1):
        image = cv2.imread(str(sample.image_path))

        if image is None:
            error = "cv2.imread returned None"
            predictions.append(make_error_prediction(config, sample, error))
            continue

        try:
            image = limit_max_side(image, max_side)
            image = PREPROCESSORS[config.preprocessing](image)

            start = time.perf_counter()
            results = predict(ocr, image, config)
            latency = time.perf_counter() - start

            lines = extract_ocr_lines(results)

            predicted, confidence = select_mrz(
                lines,
                sample.expected_mrz,
                image.shape,
            )

            exact, line_exact, distance, char_accuracy = text_metrics(
                sample.expected_mrz,
                predicted,
            )

            validation = (
                validate_check_digits(predicted) if predicted is not None else None
            )

            predictions.append(
                Prediction(
                    config_id=config.config_id,
                    stage=config.stage,
                    image_key=sample.image_key,
                    label_id=sample.label_id,
                    variant=sample.variant,
                    augmentation_types="|".join(sample.augmentation_types),
                    expected_mrz="\n".join(sample.expected_mrz),
                    predicted_mrz="\n".join(predicted or ()),
                    mrz_found=predicted is not None,
                    full_exact_match=exact,
                    line_exact_rate=line_exact,
                    char_accuracy=char_accuracy,
                    edit_distance=distance,
                    mean_mrz_confidence=confidence,
                    check_digits_valid=validation,
                    latency_seconds=latency,
                    error=None,
                )
            )

        except Exception as error:
            message = f"{type(error).__name__}: {error}"

            if printed_errors < print_errors:
                print(f"    ERROR {sample.image_key}: {message}")
                printed_errors += 1

            predictions.append(make_error_prediction(config, sample, message))

        if index == len(samples) or index % 10 == 0:
            print(f"    {index}/{len(samples)}")

    successful = [item for item in predictions if item.error is None]

    if not successful:
        first_error = next(
            (item.error for item in predictions if item.error),
            "unknown error",
        )

        return (
            {
                **asdict(config),
                "config_id": config.config_id,
                "status": "all_images_failed",
                "n_images": 0,
                "errors": len(predictions),
                "error_message": first_error,
            },
            predictions,
        )

    latencies = [item.latency_seconds for item in successful]

    exact_count = sum(item.full_exact_match for item in successful)

    found_count = sum(item.mrz_found for item in successful)

    valid_gate = [item for item in successful if item.check_digits_valid is True]

    exact_gate = sum(item.full_exact_match for item in valid_gate)

    summary = {
        **asdict(config),
        "config_id": config.config_id,
        "status": "ok",
        "n_images": len(successful),
        "errors": len(predictions) - len(successful),
        "mrz_found_rate": found_count / len(successful),
        "full_mrz_exact_match_rate": exact_count / len(successful),
        "mean_line_exact_rate": statistics.mean(
            item.line_exact_rate for item in successful
        ),
        "mean_char_accuracy": statistics.mean(
            item.char_accuracy for item in successful
        ),
        "checkdigit_gate_coverage": (len(valid_gate) / len(successful)),
        "checkdigit_gate_exact_precision": (
            exact_gate / len(valid_gate) if valid_gate else None
        ),
        "wrong_accept_rate_with_checkdigit_gate": (
            (len(valid_gate) - exact_gate) / len(valid_gate) if valid_gate else None
        ),
        "mean_latency_seconds": statistics.mean(latencies),
        "median_latency_seconds": statistics.median(latencies),
        "p95_latency_seconds": percentile(latencies, 0.95),
        "error_message": None,
    }

    return summary, predictions


def make_error_prediction(
    config: Config,
    sample: Sample,
    error: str,
) -> Prediction:
    expected = "\n".join(sample.expected_mrz)

    return Prediction(
        config_id=config.config_id,
        stage=config.stage,
        image_key=sample.image_key,
        label_id=sample.label_id,
        variant=sample.variant,
        augmentation_types="|".join(sample.augmentation_types),
        expected_mrz=expected,
        predicted_mrz="",
        mrz_found=False,
        full_exact_match=False,
        line_exact_rate=0.0,
        char_accuracy=0.0,
        edit_distance=len(expected),
        mean_mrz_confidence=None,
        check_digits_valid=None,
        latency_seconds=0.0,
        error=error,
    )


def percentile(
    values: Sequence[float],
    p: float,
) -> float:
    ordered = sorted(values)

    if not ordered:
        return 0.0

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * p
    low = math.floor(position)
    high = math.ceil(position)

    if low == high:
        return ordered[low]

    fraction = position - low

    return ordered[low] * (1 - fraction) + ordered[high] * fraction


# ============================================================================
# STAGED SEARCH
# ============================================================================


def rank_key(
    summary: dict[str, Any],
) -> tuple[float, float, float, float]:
    if summary.get("status") != "ok":
        return (-1.0, -1.0, -1.0, -math.inf)

    return (
        float(summary.get("full_mrz_exact_match_rate", 0.0)),
        float(summary.get("mean_char_accuracy", 0.0)),
        float(summary.get("mrz_found_rate", 0.0)),
        -float(summary.get("mean_latency_seconds", math.inf)),
    )


def top(
    summaries: Sequence[dict[str, Any]],
    count: int,
) -> list[dict[str, Any]]:
    return sorted(
        (item for item in summaries if item.get("status") == "ok"),
        key=rank_key,
        reverse=True,
    )[:count]


def config_from_summary(
    summary: dict[str, Any],
    stage: str,
    **overrides: Any,
) -> Config:
    values = {
        "stage": stage,
        "model_name": summary["model_name"],
        "detection_model": summary["detection_model"],
        "recognition_model": summary["recognition_model"],
        "preprocessing": summary["preprocessing"],
        "use_doc_orientation_classify": summary["use_doc_orientation_classify"],
        "use_doc_unwarping": summary["use_doc_unwarping"],
        "use_textline_orientation": summary["use_textline_orientation"],
        "text_det_thresh": summary["text_det_thresh"],
        "text_det_box_thresh": summary["text_det_box_thresh"],
        "text_det_unclip_ratio": summary["text_det_unclip_ratio"],
    }

    values.update(overrides)

    return Config(**values)


def run_stage(
    name: str,
    configs: Sequence[Config],
    samples: Sequence[Sample],
    device: str,
    max_side: int,
    output_dir: Path,
) -> tuple[list[dict[str, Any]], list[Prediction]]:
    print(
        f"\n\n################ {name}: "
        f"{len(configs)} configs x {len(samples)} images ################"
    )

    summaries: list[dict[str, Any]] = []
    predictions: list[Prediction] = []

    grouped: dict[tuple[Any, ...], list[Config]] = {}

    for config in configs:
        grouped.setdefault(
            config.ocr_key,
            [],
        ).append(config)

    for group_configs in grouped.values():
        first = group_configs[0]

        print(
            f"\nLoading OCR: {first.model_name} "
            f"(ori={first.use_doc_orientation_classify}, "
            f"uw={first.use_doc_unwarping}, "
            f"tl={first.use_textline_orientation})"
        )

        load_start = time.perf_counter()

        try:
            ocr = build_ocr(
                first,
                device,
            )

        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            print(f"  MODEL INIT FAILED: {message}")

            for config in group_configs:
                summaries.append(
                    {
                        **asdict(config),
                        "config_id": config.config_id,
                        "status": "init_failed",
                        "n_images": 0,
                        "errors": 1,
                        "error_message": message,
                    }
                )

            continue

        load_seconds = time.perf_counter() - load_start

        print(f"  loaded in {load_seconds:.2f}s")

        try:
            warmup_image = cv2.imread(str(samples[0].image_path))

            warmup_image = limit_max_side(
                warmup_image,
                max_side,
            )

            warmup_image = PREPROCESSORS[first.preprocessing](warmup_image)

            predict(
                ocr,
                warmup_image,
                first,
            )

        except Exception as error:
            print(f"  WARMUP WARNING: {type(error).__name__}: {error}")

        for config in group_configs:
            print(f"\n  >>> {config.config_id}")

            summary, items = evaluate_config(
                ocr=ocr,
                config=config,
                samples=samples,
                max_side=max_side,
            )

            summary["model_load_seconds"] = load_seconds

            summaries.append(summary)
            predictions.extend(items)

            if summary["status"] == "ok":
                print(
                    "      exact={:.1%}  "
                    "found={:.1%}  "
                    "char={:.1%}  "
                    "mean={:.3f}s  "
                    "errors={}".format(
                        summary["full_mrz_exact_match_rate"],
                        summary["mrz_found_rate"],
                        summary["mean_char_accuracy"],
                        summary["mean_latency_seconds"],
                        summary["errors"],
                    )
                )

            else:
                print(f"      FAILED: {summary.get('error_message')}")

        del ocr
        gc.collect()

    write_csv(
        output_dir / f"{name}.csv",
        sorted(
            summaries,
            key=rank_key,
            reverse=True,
        ),
    )

    return summaries, predictions


# ============================================================================
# REPORTS
# ============================================================================


def write_csv(
    path: Path,
    rows: Sequence[dict[str, Any]],
) -> None:
    if not rows:
        path.write_text(
            "",
            encoding="utf-8",
        )
        return

    fieldnames: list[str] = []
    seen: set[str] = set()

    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def augmentation_breakdown(
    predictions: Sequence[Prediction],
) -> list[dict[str, Any]]:
    groups: dict[
        tuple[str, str],
        list[Prediction],
    ] = {}

    for item in predictions:
        if item.error is not None:
            continue

        types = item.augmentation_types.split("|") or ["unknown"]

        for value in types:
            groups.setdefault(
                (
                    item.config_id,
                    value,
                ),
                [],
            ).append(item)

    rows: list[dict[str, Any]] = []

    for (
        config_id,
        augmentation,
    ), items in groups.items():
        rows.append(
            {
                "config_id": config_id,
                "augmentation_type": augmentation,
                "n_images": len(items),
                "full_mrz_exact_match_rate": statistics.mean(
                    item.full_exact_match for item in items
                ),
                "mrz_found_rate": statistics.mean(item.mrz_found for item in items),
                "mean_char_accuracy": statistics.mean(
                    item.char_accuracy for item in items
                ),
                "mean_latency_seconds": statistics.mean(
                    item.latency_seconds for item in items
                ),
            }
        )

    return rows


# ============================================================================
# MAIN
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("./dataset"),
    )

    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./outputs/benchmarks/01.ocr-recognition-sweeps"),
    )

    parser.add_argument(
        "--device",
        default="cpu",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--stage1-images",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--stage2-images",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--stage3-images",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--stage4-images",
        type=int,
        default=40,
    )

    parser.add_argument(
        "--stage5-images",
        type=int,
        default=50,
        help=(
            "Number of images used for final validation. "
            "Use 0 to run finalists on the entire dataset."
        ),
    )

    parser.add_argument(
        "--max-side",
        type=int,
        default=3000,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    samples = load_samples(
        dataset_root=args.dataset_root,
        labels_path=args.labels,
        manifest_path=args.manifest,
    )

    run_id = time.strftime("%Y%m%d_%H%M%S")

    output_dir = args.output / run_id

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Loaded {len(samples)} total images.")

    print(f"Results: {output_dir}")

    print(f"Screening resize max side: {args.max_side}px")

    all_summaries: list[dict[str, Any]] = []

    all_predictions: list[Prediction] = []

    # ------------------------------------------------------------------
    # STAGE 1
    # 6 models on small sample.
    # Keep 3.
    # ------------------------------------------------------------------

    stage1_samples = diverse_sample(
        samples,
        min(
            args.stage1_images,
            len(samples),
        ),
        args.seed,
    )

    stage1_configs = [
        Config(
            stage="stage1_models",
            model_name=model.name,
            detection_model=model.detection_model,
            recognition_model=model.recognition_model,
        )
        for model in MODELS
    ]

    stage1, predictions = run_stage(
        "stage1_models",
        stage1_configs,
        stage1_samples,
        args.device,
        args.max_side,
        output_dir,
    )

    all_summaries.extend(stage1)
    all_predictions.extend(predictions)

    winners1 = top(
        stage1,
        3,
    )

    if not winners1:
        raise RuntimeError("Every Stage 1 model failed.")

    # ------------------------------------------------------------------
    # STAGE 2
    # Top 3 models x preprocessors.
    # Keep 2.
    # ------------------------------------------------------------------

    stage2_samples = diverse_sample(
        samples,
        min(
            args.stage2_images,
            len(samples),
        ),
        args.seed + 1,
    )

    stage2_configs = [
        config_from_summary(
            winner,
            "stage2_preprocessing",
            preprocessing=preprocessing,
        )
        for winner in winners1
        for preprocessing in PREPROCESSOR_NAMES
    ]

    stage2, predictions = run_stage(
        "stage2_preprocessing",
        stage2_configs,
        stage2_samples,
        args.device,
        args.max_side,
        output_dir,
    )

    all_summaries.extend(stage2)
    all_predictions.extend(predictions)

    winners2 = top(
        stage2,
        2,
    )

    # ------------------------------------------------------------------
    # STAGE 3
    # Top 2 x useful orientation flag combinations.
    # Keep 1.
    # ------------------------------------------------------------------

    stage3_samples = diverse_sample(
        samples,
        min(
            args.stage3_images,
            len(samples),
        ),
        args.seed + 2,
    )

    stage3_configs = [
        config_from_summary(
            winner,
            "stage3_flags",
            use_doc_orientation_classify=orientation,
            use_doc_unwarping=unwarping,
            use_textline_orientation=textline,
        )
        for winner in winners2
        for (
            orientation,
            unwarping,
            textline,
        ) in PIPELINE_FLAG_SETS
    ]

    stage3, predictions = run_stage(
        "stage3_flags",
        stage3_configs,
        stage3_samples,
        args.device,
        args.max_side,
        output_dir,
    )

    all_summaries.extend(stage3)
    all_predictions.extend(predictions)

    winners3 = top(
        stage3,
        1,
    )

    # ------------------------------------------------------------------
    # STAGE 4
    # Winner x unwarping x detection parameters.
    # Keep top 2.
    # ------------------------------------------------------------------

    stage4_samples = diverse_sample(
        samples,
        min(
            args.stage4_images,
            len(samples),
        ),
        args.seed + 3,
    )

    base = winners3[0]

    stage4_configs: list[Config] = []

    for use_unwarping in (
        False,
        True,
    ):
        for (
            det_thresh,
            box_thresh,
            unclip,
        ) in DETECTION_PARAM_SETS:
            stage4_configs.append(
                config_from_summary(
                    base,
                    "stage4_final_tuning",
                    use_doc_unwarping=use_unwarping,
                    text_det_thresh=det_thresh,
                    text_det_box_thresh=box_thresh,
                    text_det_unclip_ratio=unclip,
                )
            )

    stage4, predictions = run_stage(
        "stage4_final_tuning",
        stage4_configs,
        stage4_samples,
        args.device,
        args.max_side,
        output_dir,
    )

    all_summaries.extend(stage4)
    all_predictions.extend(predictions)

    finalists = top(
        stage4,
        2,
    )

    # ------------------------------------------------------------------
    # STAGE 5
    #
    # PATCHED:
    # By default only 50 images are used.
    #
    # --stage5-images 3
    #     very fast final validation
    #
    # --stage5-images 50
    #     default
    #
    # --stage5-images 0
    #     full 357-image dataset
    # ------------------------------------------------------------------

    if args.stage5_images == 0:
        stage5_samples = list(samples)
    else:
        stage5_samples = diverse_sample(
            samples,
            min(
                args.stage5_images,
                len(samples),
            ),
            args.seed + 4,
        )

    stage5_configs = [
        config_from_summary(
            finalist,
            "stage5_final_validation",
        )
        for finalist in finalists
    ]

    stage5, predictions = run_stage(
        "stage5_final_validation",
        stage5_configs,
        stage5_samples,
        args.device,
        args.max_side,
        output_dir,
    )

    all_summaries.extend(stage5)
    all_predictions.extend(predictions)

    final_ranking = top(
        stage5,
        len(stage5),
    )

    # ------------------------------------------------------------------
    # REPORTS
    # ------------------------------------------------------------------

    write_csv(
        output_dir / "summary_all.csv",
        sorted(
            all_summaries,
            key=rank_key,
            reverse=True,
        ),
    )

    write_csv(
        output_dir / "predictions_all.csv",
        [asdict(item) for item in all_predictions],
    )

    write_csv(
        output_dir / "augmentation_breakdown.csv",
        augmentation_breakdown(all_predictions),
    )

    if final_ranking:
        with (output_dir / "best_config.json").open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                final_ranking[0],
                file,
                indent=2,
                ensure_ascii=False,
            )

    with (output_dir / "run_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            {
                "total_images": len(samples),
                "stage1_images": len(stage1_samples),
                "stage2_images": len(stage2_samples),
                "stage3_images": len(stage3_samples),
                "stage4_images": len(stage4_samples),
                "stage5_images": len(stage5_samples),
                "max_side": args.max_side,
                "seed": args.seed,
                "models": [asdict(model) for model in MODELS],
                "preprocessors": list(PREPROCESSOR_NAMES),
            },
            file,
            indent=2,
        )

    print("\n================ DONE ================")

    print(f"Results: {output_dir}")

    if final_ranking:
        best = final_ranking[0]

        print(
            "BEST: "
            f"{best['config_id']} | "
            f"exact="
            f"{best['full_mrz_exact_match_rate']:.1%} | "
            f"char="
            f"{best['mean_char_accuracy']:.1%} | "
            f"mean="
            f"{best['mean_latency_seconds']:.3f}s"
        )


if __name__ == "__main__":
    main()
