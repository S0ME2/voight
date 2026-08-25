"""Benchmark local PP-OCR recognition line batches without changing the host.

The script uses the production model factory and never installs dependencies,
starts a server, or fetches models.  Set MODEL_DIR to a directory already
containing both official Paddle model directories.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# Keep this benchmark offline even when an operator did not set the container default.
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"
DEFAULT_IMAGES = (
    ROOT / "annotation_input/passports/passport.png",
    ROOT / "annotation_input/id_cards/uzbekistan_id_001/front.png",
    ROOT / "annotation_input/id_cards/uzbekistan_id_001/back.png",
    ROOT / "annotation_input/driving_licenses/test_license_canonical.jpg",
)


@dataclass
class Measurement:
    configuration: str
    batch_size: int
    repeat: int
    status: str
    total_recognition_seconds: float | None
    lines: int
    lines_per_second: float | None
    milliseconds_per_line: float | None
    model_call_count: int
    submitted_batch_sizes: list[int]
    tensor_batch_sizes: list[int] | None
    exact_text_match_rate: float | None
    differing_lines: int | None
    score_difference_count: int | None
    error: str | None


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[16, 32, 64, 128])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--images", type=Path, nargs="+", default=list(DEFAULT_IMAGES))
    parser.add_argument("--configs", nargs="+", choices=("baseline", "hpi", "trt-fp32", "trt-fp16"))
    parser.add_argument("--output-dir", type=Path)
    parsed = parser.parse_args()
    if not parsed.batch_sizes or any(size <= 0 for size in parsed.batch_sizes):
        parser.error("--batch-sizes must contain positive integers")
    if parsed.repeats <= 0 or parsed.warmup < 0:
        parser.error("--repeats must be positive and --warmup cannot be negative")
    if any(not path.is_file() for path in parsed.images):
        parser.error("every --images path must exist")
    return parsed


def require_local_models(settings) -> None:
    root = settings.models.directory
    if root is None:
        raise RuntimeError("MODEL_DIR is required; point it at the already-cached Paddle model directory")
    missing = [
        root / "official_models" / name
        for name in (settings.models.text_detector.model, settings.models.text_recognizer.model)
        if not (root / "official_models" / name).is_dir()
    ]
    if missing:
        raise RuntimeError("required local OCR model directory is missing: " + ", ".join(map(str, missing)))


def collect_lines(settings, paths: list[Path]) -> list[Any]:
    """Detect once, then retain the same cropped line corpus for every run."""
    from app.inference.batch import _line_crop, _pad_detection_batch
    from app.models import Models

    images = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"could not decode benchmark image: {path}")
        images.append(image)
    detector = Models(settings).text_detector()
    padded = _pad_detection_batch(list(enumerate(images)))
    values = detector.detect_batch([image for _, image in padded])
    if len(values) != len(images):
        raise RuntimeError("text detector returned a different number of images")
    lines = []
    for image, result in zip(images, values):
        polygons = [region.polygon for region in result.regions]
        polygons.sort(key=lambda polygon: (float(polygon[:, 1].mean()), float(polygon[:, 0].min())))
        for polygon in polygons:
            lines.append(_line_crop(image, polygon)[0])
    if not lines:
        raise RuntimeError("text detector produced no line crops from the supplied images")
    return lines


def configuration(settings, name: str):
    runtime = settings.runtime
    options = {
        "baseline": (False, False, "fp32"),
        "hpi": (True, False, "fp32"),
        "trt-fp32": (True, True, "fp32"),
        "trt-fp16": (True, True, "fp16"),
    }[name]
    return replace(
        settings,
        runtime=replace(
            runtime,
            text_recognition_enable_hpi=options[0],
            text_recognition_use_tensorrt=options[1],
            text_recognition_precision=options[2],
        ),
    )


def recognize(model: Any, lines: list[Any], batch_size: int) -> tuple[list[tuple[str, float]], list[int], float]:
    output, sizes = [], []
    started = time.perf_counter()
    for start in range(0, len(lines), batch_size):
        batch = lines[start : start + batch_size]
        values = model.recognize_batch(batch)
        if len(values) != len(batch):
            raise RuntimeError(f"recognizer returned {len(values)} results for {len(batch)} inputs")
        sizes.append(len(batch))
        output.extend(
            (
                value.text,
                float(value.score or 0.0),
            )
            for value in values
        )
    return output, sizes, time.perf_counter() - started


def compare(actual: list[tuple[str, float]], baseline: list[tuple[str, float]]) -> tuple[float, int, int]:
    text_differences = sum(text != expected for (text, _), (expected, _) in zip(actual, baseline))
    score_differences = sum(score != expected for (_, score), (_, expected) in zip(actual, baseline))
    return (len(actual) - text_differences) / len(actual), text_differences, score_differences


def main() -> int:
    parsed = args()
    from app.config import Settings

    settings = Settings.from_env()
    if settings.runtime.text_recognition_processes != 1:
        print("benchmark preflight failed: set TEXT_RECOGNITION_PROCESSES=1", file=sys.stderr)
        return 2
    try:
        require_local_models(settings)
    except RuntimeError as error:
        print(f"benchmark preflight failed: {error}", file=sys.stderr)
        return 2
    requested = parsed.configs or (["baseline"] if settings.runtime.target == "cpu" else ["baseline", "hpi", "trt-fp32", "trt-fp16"])
    configs = ["baseline", *(name for name in requested if name != "baseline")]
    if settings.runtime.target == "cpu" and any(name != "baseline" for name in configs):
        print("benchmark preflight failed: HPI/TensorRT/FP16 configurations require RUNTIME_TARGET=gpu", file=sys.stderr)
        return 2
    try:
        lines = collect_lines(settings, parsed.images)
    except Exception as error:
        print(f"benchmark corpus setup failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2
    print(f"recognition corpus: {len(lines)} line crops")
    import paddle

    rows: list[Measurement] = []
    baselines: dict[int, list[tuple[str, float]]] = {}
    for name in configs:
        try:
            from app.models import Models

            model = Models(configuration(settings, name)).text_recognizer()
        except Exception as error:
            if name == "baseline":
                print(f"baseline initialization failed: {type(error).__name__}: {error}", file=sys.stderr)
                return 2
            for batch_size in parsed.batch_sizes:
                rows.append(Measurement(name, batch_size, 0, "unsupported", None, len(lines), None, None, 0, [], None, None, None, None, f"{type(error).__name__}: {error}"))
            continue
        for batch_size in parsed.batch_sizes:
            try:
                for _ in range(parsed.warmup):
                    recognize(model, lines, batch_size)
                for repeat in range(1, parsed.repeats + 1):
                    values, submitted, elapsed = recognize(model, lines, batch_size)
                    if name == "baseline":
                        baselines.setdefault(batch_size, values)
                    match_rate, differences, score_differences = compare(values, baselines[batch_size])
                    rows.append(Measurement(name, batch_size, repeat, "ok", elapsed, len(lines), len(lines) / elapsed, 1000 * elapsed / len(lines), len(submitted), submitted, None, match_rate, differences, score_differences, None))
            except Exception as error:
                resource_error = any(marker in str(error).lower() for marker in ("memory", "resource exhausted", "cuda error"))
                rows.append(Measurement(name, batch_size, 0, "resource_exhausted" if resource_error else "failed", None, len(lines), None, None, 0, [], None, None, None, None, f"{type(error).__name__}: {error}"))
    output = parsed.output_dir or ROOT / "benchmarks/results" / f"recognition-{datetime.now():%Y%m%dT%H%M%S}"
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_target": settings.runtime.target,
        "paddle_version": paddle.__version__,
        "selected_device": "cpu" if settings.runtime.target == "cpu" else f"gpu:{settings.runtime.gpu_id}",
        "recognition_model": settings.models.text_recognizer.model,
        "line_count": len(lines),
        "tensor_batch_sizes": "unavailable: Paddle TextRecognition does not expose its internal tensor shape",
    }
    (output / "recognition_batch_benchmark.json").write_text(json.dumps({"metadata": metadata, "rows": [asdict(row) for row in rows]}, indent=2), encoding="utf-8")
    with (output / "recognition_batch_benchmark.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=Measurement.__annotations__)
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    successful = [row for row in rows if row.status == "ok"]
    for row in successful:
        print(f"{row.configuration:9} batch={row.batch_size:3} repeat={row.repeat} {row.lines_per_second:.2f} lines/s {row.milliseconds_per_line:.2f} ms/line match={row.exact_text_match_rate:.3f}")
    if successful:
        best = max(successful, key=lambda row: row.lines_per_second or 0)
        print(f"best measured: {best.configuration} batch={best.batch_size} ({best.lines_per_second:.2f} lines/s)")
    print(output)
    return 0 if successful else 1


if __name__ == "__main__":
    raise SystemExit(main())
