from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import numpy as np
from paddleocr import PaddleOCR


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MRZ_ALLOWED = re.compile(r"^[A-Z0-9<]+$")


@dataclass(frozen=True)
class Sample:
    image_path: Path
    label_id: str
    expected_mrz: tuple[str, ...]


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
    contrast: float
    image: str
    expected_mrz: str
    predicted_mrz: str
    mrz_found: bool
    full_exact_match: bool
    line_exact_rate: float
    char_accuracy: float
    edit_distance: int
    mean_confidence: float | None
    latency_seconds: float
    error: str | None


def normalize(text: str) -> str:
    return "".join(str(text).upper().split())


def load_samples(dataset_root: Path) -> list[Sample]:
    labels_path = dataset_root / "labels.json"
    source_dir = dataset_root / "source"

    with labels_path.open("r", encoding="utf-8") as file:
        raw_labels = json.load(file)

    labels: dict[str, tuple[str, ...]] = {}

    for filename, value in raw_labels.items():
        mrz = value["mrz"] if isinstance(value, dict) else value
        labels[filename] = tuple(normalize(line) for line in mrz)

    samples: list[Sample] = []

    for image_path in sorted(source_dir.iterdir()):
        if not image_path.is_file():
            continue

        if image_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        if image_path.name not in labels:
            print(f"WARNING: no label for {image_path.name}; skipping")
            continue

        samples.append(
            Sample(
                image_path=image_path,
                label_id=image_path.name,
                expected_mrz=labels[image_path.name],
            )
        )

    if not samples:
        raise RuntimeError("No labeled source images found.")

    return samples


def resize_max_side(image: np.ndarray, max_side: int) -> np.ndarray:
    height, width = image.shape[:2]
    longest = max(height, width)

    if longest <= max_side:
        return image

    scale = max_side / longest

    return cv2.resize(
        image,
        (
            max(1, int(round(width * scale))),
            max(1, int(round(height * scale))),
        ),
        interpolation=cv2.INTER_AREA,
    )


def adjust_contrast(gray: np.ndarray, factor: float) -> np.ndarray:
    """
    factor = 1.0 -> current grayscale baseline
    factor > 1.0 -> stronger contrast

    Contrast is adjusted around the image mean so this experiment does not
    intentionally change overall brightness.
    """
    if factor <= 0:
        raise ValueError("Contrast factor must be > 0.")

    if factor == 1.0:
        return gray.copy()

    mean = float(gray.mean())

    adjusted = (gray.astype(np.float32) - mean) * factor + mean

    return np.clip(adjusted, 0, 255).astype(np.uint8)


def preprocess(
    image: np.ndarray,
    contrast: float,
    max_side: int,
) -> np.ndarray:
    image = resize_max_side(image, max_side)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = adjust_contrast(gray, contrast)

    # Keep the same 3-channel grayscale format used by the winning pipeline.
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def create_ocr() -> PaddleOCR:
    return PaddleOCR(
        text_detection_model_name="PP-OCRv6_medium_det",
        text_recognition_model_name="PP-OCRv6_medium_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )


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

            nested = data.get("res")

            if isinstance(nested, dict):
                return nested.get(key, default)
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
        texts = to_list(result_field(result, "rec_texts", []))
        scores = to_list(result_field(result, "rec_scores", []))
        boxes = to_list(result_field(result, "rec_boxes", []))

        if len(boxes) != len(texts):
            boxes = [
                [0, i * 20, max(10, len(str(text)) * 12), i * 20 + 16]
                for i, text in enumerate(texts)
            ]

        for i, text in enumerate(texts):
            cleaned = normalize(str(text))

            if not cleaned:
                continue

            score = float(scores[i]) if i < len(scores) else 0.0
            flat = np.asarray(boxes[i]).reshape(-1)

            if flat.size >= 4:
                x1, y1, x2, y2 = map(float, flat[:4])
            else:
                x1, y1, x2, y2 = 0.0, i * 20.0, 100.0, i * 20.0 + 16.0

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


def charset_ratio(text: str) -> float:
    if not text:
        return 0.0

    return sum(
        char == "<" or char.isdigit() or ("A" <= char <= "Z") for char in text
    ) / len(text)


def merge_same_row(lines: Sequence[OCRLine]) -> list[OCRLine]:
    clusters: list[list[OCRLine]] = []

    for line in sorted(lines, key=lambda item: item.center_y):
        target = None

        for cluster in clusters:
            center = statistics.mean(item.center_y for item in cluster)
            height = statistics.median(item.height for item in cluster)

            if abs(line.center_y - center) <= max(height, line.height) * 0.65:
                target = cluster
                break

        if target is None:
            clusters.append([line])
        else:
            target.append(line)

    merged: list[OCRLine] = []

    for cluster in clusters:
        ordered = sorted(cluster, key=lambda item: item.x1)
        text = "".join(item.text for item in ordered)
        total_weight = sum(max(1, len(item.text)) for item in ordered)

        score = (
            sum(item.score * max(1, len(item.text)) for item in ordered) / total_weight
        )

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
        + 3.0 * charset_ratio(line.text)
        + (1.0 if "<" in line.text else 0.0)
        + 0.5 * min(1.0, line.width / max(1.0, image_width * 0.5))
        + 0.25 * min(1.0, line.center_y / max(1.0, image_height))
    )


def select_mrz(
    lines: Sequence[OCRLine],
    expected: Sequence[str],
    image_shape: tuple[int, ...],
) -> tuple[tuple[str, ...] | None, float | None]:
    height, width = image_shape[:2]
    target_lengths = tuple(len(line) for line in expected)
    target_count = len(target_lengths)

    candidates = [
        line
        for line in list(lines) + merge_same_row(lines)
        if len(line.text) >= 15 and charset_ratio(line.text) >= 0.65
    ]

    if len(candidates) < target_count:
        return None, None

    median_length = int(statistics.median(target_lengths))

    candidates = sorted(
        candidates,
        key=lambda item: candidate_score(
            item,
            median_length,
            width,
            height,
        ),
        reverse=True,
    )[:12]

    best: tuple[OCRLine, ...] | None = None
    best_score = -math.inf

    for combo in itertools.combinations(candidates, target_count):
        ordered = tuple(sorted(combo, key=lambda item: item.center_y))

        # Avoid selecting two OCR fragments from the same physical row.
        if any(
            abs(ordered[i + 1].center_y - ordered[i].center_y)
            < 0.25 * max(ordered[i].height, ordered[i + 1].height)
            for i in range(len(ordered) - 1)
        ):
            continue

        score = sum(
            candidate_score(
                line,
                target_lengths[i],
                width,
                height,
            )
            for i, line in enumerate(ordered)
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


def levenshtein(left: str, right: str) -> int:
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


def calculate_metrics(
    expected: Sequence[str],
    predicted: Sequence[str] | None,
) -> tuple[bool, float, int, float]:
    expected_tuple = tuple(expected)
    predicted_tuple = tuple(predicted or ())

    exact = expected_tuple == predicted_tuple

    line_exact = sum(
        i < len(predicted_tuple) and predicted_tuple[i] == expected_line
        for i, expected_line in enumerate(expected_tuple)
    ) / max(1, len(expected_tuple))

    expected_text = "\n".join(expected_tuple)
    predicted_text = "\n".join(predicted_tuple)

    distance = levenshtein(expected_text, predicted_text)

    char_accuracy = max(
        0.0,
        1.0 - distance / max(1, len(expected_text), len(predicted_text)),
    )

    return exact, line_exact, distance, char_accuracy


def run_prediction(
    ocr: PaddleOCR,
    image: np.ndarray,
) -> list[Any]:
    return list(
        ocr.predict(
            image,
            text_det_thresh=0.30,
            text_det_box_thresh=0.50,
            text_det_unclip_ratio=2.00,
            text_rec_score_thresh=0.0,
        )
    )


def benchmark_one_contrast(
    ocr: PaddleOCR,
    samples: Sequence[Sample],
    contrast: float,
    max_side: int,
) -> tuple[dict[str, Any], list[Prediction]]:
    predictions: list[Prediction] = []

    print(f"\n===== Contrast {contrast:.2f}x =====")

    for index, sample in enumerate(samples, start=1):
        try:
            image = cv2.imread(str(sample.image_path))

            if image is None:
                raise ValueError("Could not read image.")

            processed = preprocess(
                image=image,
                contrast=contrast,
                max_side=max_side,
            )

            start = time.perf_counter()
            results = run_prediction(ocr, processed)
            latency = time.perf_counter() - start

            ocr_lines = extract_ocr_lines(results)

            predicted, confidence = select_mrz(
                ocr_lines,
                sample.expected_mrz,
                processed.shape,
            )

            exact, line_exact, distance, char_accuracy = calculate_metrics(
                sample.expected_mrz,
                predicted,
            )

            predictions.append(
                Prediction(
                    contrast=contrast,
                    image=sample.image_path.name,
                    expected_mrz="\n".join(sample.expected_mrz),
                    predicted_mrz="\n".join(predicted or ()),
                    mrz_found=predicted is not None,
                    full_exact_match=exact,
                    line_exact_rate=line_exact,
                    char_accuracy=char_accuracy,
                    edit_distance=distance,
                    mean_confidence=confidence,
                    latency_seconds=latency,
                    error=None,
                )
            )

        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            print(f"  ERROR {sample.image_path.name}: {message}")

            predictions.append(
                Prediction(
                    contrast=contrast,
                    image=sample.image_path.name,
                    expected_mrz="\n".join(sample.expected_mrz),
                    predicted_mrz="",
                    mrz_found=False,
                    full_exact_match=False,
                    line_exact_rate=0.0,
                    char_accuracy=0.0,
                    edit_distance=len("\n".join(sample.expected_mrz)),
                    mean_confidence=None,
                    latency_seconds=0.0,
                    error=message,
                )
            )

        print(f"  {index}/{len(samples)}")

    successful = [item for item in predictions if item.error is None]

    if not successful:
        return (
            {
                "contrast": contrast,
                "status": "all_failed",
                "n_images": 0,
                "errors": len(predictions),
            },
            predictions,
        )

    summary = {
        "contrast": contrast,
        "status": "ok",
        "n_images": len(successful),
        "errors": len(predictions) - len(successful),
        "mrz_found_rate": statistics.mean(item.mrz_found for item in successful),
        "full_mrz_exact_match_rate": statistics.mean(
            item.full_exact_match for item in successful
        ),
        "mean_line_exact_rate": statistics.mean(
            item.line_exact_rate for item in successful
        ),
        "mean_char_accuracy": statistics.mean(
            item.char_accuracy for item in successful
        ),
        "mean_latency_seconds": statistics.mean(
            item.latency_seconds for item in successful
        ),
        "median_latency_seconds": statistics.median(
            item.latency_seconds for item in successful
        ),
    }

    print(
        "  exact={:.1%} | char={:.1%} | found={:.1%} | mean={:.3f}s".format(
            summary["full_mrz_exact_match_rate"],
            summary["mean_char_accuracy"],
            summary["mrz_found_rate"],
            summary["mean_latency_seconds"],
        )
    )

    return summary, predictions


def rank_key(summary: dict[str, Any]) -> tuple[float, float, float, float]:
    if summary.get("status") != "ok":
        return (-1.0, -1.0, -1.0, -math.inf)

    return (
        float(summary["full_mrz_exact_match_rate"]),
        float(summary["mean_char_accuracy"]),
        float(summary["mrz_found_rate"]),
        -float(summary["mean_latency_seconds"]),
    )


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames: list[str] = []

    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("./dataset"),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./outputs/benchmarks/02.contrast-sweep"),
    )

    parser.add_argument(
        "--max-side",
        type=int,
        default=3000,
    )

    parser.add_argument(
        "--contrasts",
        default="1.0,1.1,1.25,1.5,1.75,2.0,2.5,3.0",
        help=(
            "Comma-separated contrast factors. 1.0 is your current grayscale baseline."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    contrasts = [
        float(item.strip()) for item in args.contrasts.split(",") if item.strip()
    ]

    samples = load_samples(args.dataset_root)

    output_dir = args.output / time.strftime("%Y%m%d_%H%M%S")

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Loaded {len(samples)} original source images.")
    print(f"Contrasts: {contrasts}")
    print(f"Output: {output_dir}")

    print("\nLoading winning PaddleOCR model once...")
    ocr = create_ocr()

    # Warm up the model once before timing.
    warmup_image = cv2.imread(str(samples[0].image_path))

    if warmup_image is not None:
        try:
            warmup_image = preprocess(
                warmup_image,
                contrast=1.0,
                max_side=args.max_side,
            )
            run_prediction(ocr, warmup_image)
        except Exception as error:
            print(f"Warmup warning: {error}")

    summaries: list[dict[str, Any]] = []
    predictions: list[Prediction] = []

    for contrast in contrasts:
        summary, items = benchmark_one_contrast(
            ocr=ocr,
            samples=samples,
            contrast=contrast,
            max_side=args.max_side,
        )

        summaries.append(summary)
        predictions.extend(items)

    ranking = sorted(
        summaries,
        key=rank_key,
        reverse=True,
    )

    write_csv(
        output_dir / "contrast_ranking.csv",
        ranking,
    )

    write_csv(
        output_dir / "contrast_predictions.csv",
        [asdict(item) for item in predictions],
    )

    with (output_dir / "best_contrast.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            ranking[0],
            file,
            indent=2,
        )

    print("\n================ FINAL RANKING ================")

    for index, result in enumerate(ranking, start=1):
        if result.get("status") != "ok":
            continue

        print(
            f"{index}. contrast={result['contrast']:.2f}x | "
            f"exact={result['full_mrz_exact_match_rate']:.1%} | "
            f"char={result['mean_char_accuracy']:.1%} | "
            f"found={result['mrz_found_rate']:.1%} | "
            f"mean={result['mean_latency_seconds']:.3f}s"
        )

    print(f"\nResults saved to: {output_dir}")
    print("1.00x is your current grayscale baseline.")


if __name__ == "__main__":
    main()
