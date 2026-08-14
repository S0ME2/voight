"""Benchmark batched MRZ detection followed by Paddle OCR on ID-card backs."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class Card:
    card_id: str
    front: Path
    back: Path


@dataclass
class Measurement:
    batch_size: int
    repeat: int
    status: str
    total_seconds: float | None
    localization_seconds: float | None
    text_detection_seconds: float | None
    text_recognition_seconds: float | None
    source_card_ids: list[str]
    mrz_detected: int
    ocr_lines: int
    recognized_mrz: dict[str, list[str]]
    localization_tensor_batch_sizes: list[int]
    detection_tensor_batch_sizes: list[int]
    recognition_tensor_batch_sizes: list[int]
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id-card-dir", type=Path, default=ROOT / "dataset/id_card")
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.max_batch_size < 1 or args.max_batch_size & (args.max_batch_size - 1):
        parser.error("--max-batch-size must be a positive power of two")
    if args.repeats < 1 or args.warmup < 0:
        parser.error("--repeats must be positive and --warmup cannot be negative")
    args.cards = load_cards(args.id_card_dir, parser)
    args.sizes = [1 << power for power in range(args.max_batch_size.bit_length())]
    return args


def load_cards(directory: Path, parser: argparse.ArgumentParser | None = None) -> list[Card]:
    def fail(message: str):
        if parser:
            parser.error(message)
        raise ValueError(message)

    if not directory.is_dir():
        fail(f"ID-card directory does not exist: {directory}")
    cards = []
    for card_dir in sorted(path for path in directory.iterdir() if path.is_dir()):
        sides = {}
        for side in ("front", "back"):
            matches = sorted(
                path for path in card_dir.iterdir()
                if path.is_file() and path.stem.lower() == side and path.suffix.lower() in IMAGE_EXTENSIONS
            )
            if len(matches) != 1:
                fail(f"{card_dir} must contain exactly one {side} image")
            sides[side] = matches[0]
        cards.append(Card(card_dir.name, sides["front"], sides["back"]))
    if not cards:
        fail(f"no ID-card directories found in {directory}")
    return cards


def workload(cards: list[Card], size: int) -> list[Card]:
    return [cards[index % len(cards)] for index in range(size)]


def tensor_sizes(stage: Any) -> list[int]:
    return [int(size) for size in stage.get("tensor_batch_sizes", [])] if isinstance(stage, dict) else []


def measure(models: Any, settings: Any, cards: list[Card], size: int, repeat: int) -> Measurement:
    from app.documents.mrz import ID_CARD, crop_polygon, preprocess, reconstruct, select
    from app.inference.batch import BatchedOcr, OcrSample

    selected = workload(cards, size)
    source_ids = [card.card_id for card in selected]
    images = [cv2.imread(str(card.back)) for card in selected]
    if any(image is None for image in images):
        return Measurement(size, repeat, "failed", None, None, None, None, source_ids, 0, 0, {}, [], [], [], "could not decode an ID-card back")
    started = time.perf_counter()
    detected = 0
    try:
        localizer = models.mrz_localizer()
        localization_started = time.perf_counter()
        localized = localizer.localize_batch(images)
        localization_seconds = time.perf_counter() - localization_started
        crops = []
        for image, result in zip(images, localized):
            if result.polygon.size < 8:
                continue
            detected += 1
            crop = crop_polygon(image, result.polygon, settings.mrz.polygon_padding_ratio)[0]
            crops.append(preprocess(crop, settings.mrz.max_side, settings.mrz.contrast))
        if len(crops) != size:
            raise RuntimeError(f"MRZ detector found {len(crops)}/{size} usable polygons")
        ocr = BatchedOcr(
            models.text_detector(),
            models.text_recognizer(),
            detection_batch_size=size,
            recognition_batch_size=max(size, size * 3),
        )
        result = ocr.run([OcrSample(str(index), crop) for index, crop in enumerate(crops)])
        diagnostics = result.diagnostics
        text_detection_seconds = diagnostics["text_detection"]["wall_seconds"]
        text_recognition_seconds = diagnostics["text_recognition"]["wall_seconds"]
        recognized = {}
        line_count = 0
        for index, card in enumerate(selected):
            lines = select(reconstruct(result.tokens.get(str(index), [])), ID_CARD.line_counts)
            recognized.setdefault(card.card_id, []).append("\n".join(line.text for line in lines))
            line_count += len(lines)
        return Measurement(
            size, repeat, "ok", time.perf_counter() - started, localization_seconds,
            text_detection_seconds, text_recognition_seconds,
            source_ids, detected, line_count, recognized,
            [int(localizer.last_tensor_batch_size)],
            tensor_sizes(diagnostics["text_detection"]),
            tensor_sizes(diagnostics["text_recognition"]),
            None,
        )
    except Exception as error:
        return Measurement(size, repeat, "failed", time.perf_counter() - started, None, None, None, source_ids, detected, 0, {}, [], [], [], f"{type(error).__name__}: {error}")


def summarize(rows: list[Measurement]) -> list[dict[str, Any]]:
    summary = []
    for size in sorted({row.batch_size for row in rows}):
        values = [row for row in rows if row.batch_size == size and row.status == "ok"]
        if not values:
            continue
        summary.append({
            "batch_size": size,
            "repeats": len(values),
            "median_seconds": statistics.median(row.total_seconds for row in values),
            "median_text_detection_seconds": statistics.median(row.text_detection_seconds for row in values),
            "median_text_recognition_seconds": statistics.median(row.text_recognition_seconds for row in values),
            "cards_per_second": size / statistics.median(row.total_seconds for row in values),
            "mrz_detected": sum(row.mrz_detected for row in values),
            "ocr_lines": sum(row.ocr_lines for row in values),
            "localization_tensor_batch_sizes": values[0].localization_tensor_batch_sizes,
            "detection_tensor_batch_sizes": values[0].detection_tensor_batch_sizes,
            "recognition_tensor_batch_sizes": values[0].recognition_tensor_batch_sizes,
            "source_card_ids": values[0].source_card_ids,
        })
    return summary


def plot(path: Path, summary: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sizes = [row["batch_size"] for row in summary]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(sizes, [row["median_seconds"] for row in summary], marker="o", label="total")
    axes[0].plot(sizes, [row["median_text_detection_seconds"] for row in summary], marker="o", label="text detection")
    axes[0].plot(sizes, [row["median_text_recognition_seconds"] for row in summary], marker="o", label="text recognition")
    axes[0].set(xlabel="ID cards per batch", ylabel="Median seconds", title="MRZ + Paddle OCR time")
    axes[0].legend()
    axes[1].plot(sizes, [row["cards_per_second"] for row in summary], marker="o", color="tab:green")
    axes[1].set(xlabel="ID cards per batch", ylabel="Cards per second", title="Throughput")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    from app.config import Settings
    from app.models import Models

    settings = Settings.from_env()
    if settings.runtime.target != "cpu":
        print("set RUNTIME_TARGET=cpu for local benchmarking", file=sys.stderr)
        return 2
    if settings.runtime.text_recognition_processes != 1:
        print("set TEXT_RECOGNITION_PROCESSES=1 for this benchmark", file=sys.stderr)
        return 2
    if settings.models.directory is None:
        print("MODEL_DIR must point to the local model cache", file=sys.stderr)
        return 2
    models = Models(settings)
    for _ in range(args.warmup):
        warmup = measure(models, settings, args.cards, 1, 0)
        if warmup.status != "ok":
            print(f"warm-up failed: {warmup.error}", file=sys.stderr)
            return 2
    rows = [
        measure(models, settings, args.cards, size, repeat)
        for size in args.sizes
        for repeat in range(1, args.repeats + 1)
    ]
    summary = summarize(rows)
    output = args.output_dir or ROOT / "complexity/results" / f"mrz-paddle-{datetime.now():%Y%m%dT%H%M%S}"
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "run": {
            "runtime": settings.runtime.target,
            "id_card_source_dir": str(args.id_card_dir),
            "id_card_source_cards": [card.card_id for card in args.cards],
            "mrz_input_side": "back",
            "sizes": args.sizes,
            "repeats": args.repeats,
            "summary_statistic": "median",
            "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "rows": [asdict(row) for row in rows],
        "summary": summary,
    }
    (output / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "benchmark.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=Measurement.__annotations__)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    plot(output / "benchmark.png", summary)
    for row in summary:
        print(
            f"N={row['batch_size']:2} total={row['median_seconds']:.3f}s "
            f"detection={row['median_text_detection_seconds']:.3f}s "
            f"recognition={row['median_text_recognition_seconds']:.3f}s "
            f"throughput={row['cards_per_second']:.3f} cards/s"
        )
    print(output)
    return 0 if len(rows) == len(args.sizes) * args.repeats and all(row.status == "ok" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
