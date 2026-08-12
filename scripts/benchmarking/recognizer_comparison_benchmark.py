"""Compare locally cached text recognizers and crop-packing strategies on CPU."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"

DEFAULT_MODELS = (
    "PP-OCRv6_medium_rec",
    "PP-OCRv6_small_rec",
    "latin_PP-OCRv5_mobile_rec",
)


@dataclass
class Row:
    backend: str
    model: str
    model_path: str
    packing_strategy: str
    batch_size: int
    status: str
    total_lines: int
    total_seconds: float | None
    lines_per_second: float | None
    milliseconds_per_line: float | None
    actual_batch_sizes: list[int]
    mean_padding_efficiency: float | None
    exact_text_match_against_truth: float | None
    character_accuracy: float | None
    differing_outputs_from_production_baseline: int | None
    confidence_difference_count: int | None
    failure_count: int
    error: str | None


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--packing", nargs="+", choices=("sequential", "aspect-ratio"), default=["sequential", "aspect-ratio"])
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 1 for size in args.batch_sizes):
        parser.error("--repeats and --batch-sizes must be positive")
    return args


def run(model, lines, packer, batch_size):
    outputs, sizes, efficiencies = {}, [], []
    started = time.perf_counter()
    for batch in packer.pack(list(enumerate(lines)), batch_size):
        images = [image for _, image in batch]
        values = model.recognize_batch(images)
        if len(values) != len(batch):
            raise ValueError("recognizer returned a different result count")
        outputs.update((index, value) for (index, _), value in zip(batch, values))
        sizes.append(len(batch))
        ratios = [image.shape[1] / max(1, image.shape[0]) for image in images]
        efficiencies.append(sum(ratios) / (max(ratios) * len(ratios)))
    return [outputs[index] for index in range(len(lines))], sizes, efficiencies, time.perf_counter() - started


def main() -> int:
    args = arguments()
    from app.config import Settings
    from app.inference.packing import recognition_batch_packer
    from app.models import Models
    from scripts.benchmarking.text_recognition_batch_benchmark import collect_lines

    settings = Settings.from_env()
    if settings.runtime.target != "cpu" or settings.runtime.text_recognition_processes != 1:
        print("set RUNTIME_TARGET=cpu and TEXT_RECOGNITION_PROCESSES=1", file=sys.stderr)
        return 2
    root = settings.models.directory
    if root is None:
        raise RuntimeError("MODEL_DIR is required; point it at a prepared Paddle model directory")
    detector_path = root / "official_models" / settings.models.text_detector.model
    if not detector_path.is_dir():
        print(f"local detector is unavailable: {detector_path}", file=sys.stderr)
        return 2
    settings = replace(settings, models=replace(settings.models, directory=root))
    lines = collect_lines(settings, [
        ROOT / "annotation_input/passports/passport.png",
        ROOT / "annotation_input/id_cards/uzbekistan_id_001/front.png",
        ROOT / "annotation_input/id_cards/uzbekistan_id_001/back.png",
        ROOT / "annotation_input/driving_licenses/test_license_canonical.jpg",
    ])
    production_model = settings.models.text_recognizer.model
    models = [production_model, *(name for name in args.models if name != production_model)]
    rows, baselines = [], {}
    for name in models:
        model_path = root / "official_models" / name
        if not model_path.is_dir():
            for strategy in args.packing:
                for batch_size in args.batch_sizes:
                    rows.append(Row("paddle", name, str(model_path), strategy, batch_size, "unavailable", len(lines), None, None, None, [], None, None, None, None, None, len(lines), "local weights unavailable"))
            continue
        selected = replace(settings, models=replace(settings.models, text_recognizer=replace(settings.models.text_recognizer, model=name)))
        try:
            recognizer = Models(selected).text_recognizer()
        except Exception as error:
            for strategy in args.packing:
                for batch_size in args.batch_sizes:
                    rows.append(Row("paddle", name, str(model_path), strategy, batch_size, "failed", len(lines), None, None, None, [], None, None, None, None, None, len(lines), f"{type(error).__name__}: {error}"))
            continue
        for strategy in args.packing:
            packer = recognition_batch_packer(strategy)
            for batch_size in args.batch_sizes:
                for _ in range(args.repeats):
                    try:
                        values, sizes, efficiencies, elapsed = run(recognizer, lines, packer, batch_size)
                        key = batch_size
                        if name == production_model and strategy == "sequential":
                            baselines.setdefault(key, values)
                        baseline = baselines.get(key)
                        text_diff = None if baseline is None else sum(value.text != expected.text for value, expected in zip(values, baseline))
                        score_diff = None if baseline is None else sum(value.score != expected.score for value, expected in zip(values, baseline))
                        rows.append(Row("paddle", name, str(model_path), strategy, batch_size, "ok", len(lines), elapsed, len(lines) / elapsed, elapsed * 1000 / len(lines), sizes, sum(efficiencies) / len(efficiencies), None, None, text_diff, score_diff, 0, None))
                    except Exception as error:
                        rows.append(Row("paddle", name, str(model_path), strategy, batch_size, "failed", len(lines), None, None, None, [], None, None, None, None, None, len(lines), f"{type(error).__name__}: {error}"))
    output = args.output_dir or ROOT / "complexity/results" / f"recognizers-{datetime.now():%Y%m%dT%H%M%S}"
    output.mkdir(parents=True, exist_ok=True)
    report = {"runtime": "cpu", "truth_available": False, "production_baseline_model": production_model, "total_lines": len(lines), "rows": [asdict(row) for row in rows]}
    (output / "recognizer_comparison.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (output / "recognizer_comparison.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=Row.__annotations__)
        writer.writeheader(); writer.writerows(asdict(row) for row in rows)
    for row in rows:
        speed = "" if row.lines_per_second is None else f" {row.lines_per_second:.2f} lines/s"
        print(f"{row.model} {row.packing_strategy} batch={row.batch_size}: {row.status}{speed}")
    print(output)
    return 1 if any(row.status == "failed" for row in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
