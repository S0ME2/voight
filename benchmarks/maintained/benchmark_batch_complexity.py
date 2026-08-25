"""Measure total batch time and real model batching on production ``/v1`` routes.

``--max-batch-size 16`` runs 1, 2, 4, 8, and 16. GPU mode runs only on the
V100 server; this script never imports, installs, or initializes GPU libraries.
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import requests

ROOT = Path(__file__).resolve().parents[2]
SAMPLES = ROOT / "annotation_input"
KINDS = ("passport", "id-card", "driving-license")
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IdCard = tuple[str, Path, Path]


@dataclass
class Measurement:
    runtime: str
    document_type: str
    document_count: int
    repeat: int
    client_total_seconds: float
    server_total_seconds: float | None
    status: str
    failure_detail: str | None
    localization_tensor_batches: dict[str, list[int]]
    detection_tensor_batches: list[int]
    recognition_tensor_batches: list[int]
    detected_line_count: int | None
    recognition_candidate_count: int | None
    filtered_before_recognition_count: int | None
    recognition_share_of_total: float | None
    stage_seconds: dict[str, float]
    mrz_recognition_tensor_batches: list[int] | None = None
    id_card_front_fallback_scan_count: int | None = None
    recognition_packing_strategy: str | None = None
    succeeded: int | None = None
    failed: int | None = None
    source_card_ids: tuple[str, ...] = ()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--runtime", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--document-type", choices=("all", *KINDS), default="all")
    parser.add_argument("--max-batch-size", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--no-warmup", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--id-card-dir", type=Path, help="Directory containing one front.* and back.* pair per card")
    parser.add_argument("--passport", type=Path, default=SAMPLES / "passports/passport.png")
    parser.add_argument("--id-front", type=Path, default=SAMPLES / "id_cards/uzbekistan_id_001/front.png")
    parser.add_argument("--id-back", type=Path, default=SAMPLES / "id_cards/uzbekistan_id_001/back.png")
    parser.add_argument("--driving-license", type=Path, default=SAMPLES / "driving_licenses/test_license_canonical.jpg")
    args = parser.parse_args()
    if args.max_batch_size < 1 or args.max_batch_size & (args.max_batch_size - 1):
        parser.error("--max-batch-size must be a positive power of two")
    if args.repeats < 1 or args.timeout <= 0:
        parser.error("--repeats and --timeout must be positive")
    args.kinds = KINDS if args.document_type == "all" else (args.document_type,)
    if "id-card" in args.kinds and args.id_card_dir:
        try:
            args.id_cards = load_id_cards(args.id_card_dir)
        except ValueError as error:
            parser.error(str(error))
    else:
        args.id_cards = [("cli-card", args.id_front, args.id_back)]
    required_paths = []
    if "passport" in args.kinds:
        required_paths.append(args.passport)
    if "driving-license" in args.kinds:
        required_paths.append(args.driving_license)
    if "id-card" in args.kinds:
        required_paths.extend((args.id_cards[0][1], args.id_cards[0][2]))
    if any(not path.is_file() for path in required_paths):
        parser.error("all sample paths must exist")
    args.sizes = [1 << power for power in range(args.max_batch_size.bit_length())]
    return args


def load_id_cards(directory: Path) -> list[IdCard]:
    if not directory.is_dir():
        raise ValueError(f"ID-card directory does not exist: {directory}")
    cards = []
    for card_dir in sorted(path for path in directory.iterdir() if path.is_dir()):
        sides = {}
        for side in ("front", "back"):
            matches = sorted(
                path for path in card_dir.iterdir()
                if path.is_file() and path.stem.lower() == side and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
            )
            if len(matches) != 1:
                raise ValueError(f"{card_dir} must contain exactly one {side} image")
            sides[side] = matches[0]
        cards.append((card_dir.name, sides["front"], sides["back"]))
    if not cards:
        raise ValueError(f"no ID-card directories found in {directory}")
    return cards


def id_archive(cards: list[IdCard], count: int) -> tuple[bytes, tuple[str, ...]]:
    output = BytesIO()
    card_ids = []
    with zipfile.ZipFile(output, "w") as archive:
        for index in range(count):
            card_id, front, back = cards[index % len(cards)]
            card_ids.append(card_id)
            for side, path in (("front", front), ("back", back)):
                archive.writestr(f"card-{index:03d}/{side}{path.suffix.lower()}", path.read_bytes())
    return output.getvalue(), tuple(card_ids)


def tensor_batches(stage: Any) -> dict[str, list[int]]:
    if not isinstance(stage, dict):
        return {}
    if "tensor_batch_sizes" in stage:
        return {"default": [int(value) for value in stage["tensor_batch_sizes"]]}
    return {str(name): [int(size) for size in value.get("tensor_batch_sizes", [])] for name, value in stage.items() if isinstance(value, dict) and "tensor_batch_sizes" in value}


def stage_seconds(diagnostics: dict[str, Any], total: float) -> dict[str, float]:
    pipeline = diagnostics.get("pipeline", {})
    values = {
        "localization": sum(float(stage.get("wall_seconds", 0.0)) for stage in diagnostics.get("localization", {}).values() if isinstance(stage, dict)),
        "canonicalization": float(pipeline.get("canonicalization_seconds", 0.0)),
        "data_crop": float(pipeline.get("data_crop_seconds", 0.0)),
        "mrz_crop_preprocess": float(pipeline.get("mrz_crop_preprocess_seconds", 0.0)),
        "text_detection": float(diagnostics.get("text_detection", {}).get("wall_seconds", 0.0)),
        "text_line_crops": float(diagnostics.get("line_crop_seconds", 0.0)),
        "text_recognition": float(diagnostics.get("text_recognition", {}).get("wall_seconds", 0.0)),
        "mrz_recognition": float(diagnostics.get("mrz_recognition", {}).get("wall_seconds", 0.0)),
        "ocr_result_unpack": float(diagnostics.get("result_unpack_seconds", 0.0)),
        "result_assembly": float(pipeline.get("result_assembly_seconds", 0.0)),
    }
    values["other_request_work"] = max(0.0, total - sum(values.values()))
    return values


def post(args: argparse.Namespace, kind: str, count: int, repeat: int) -> Measurement:
    url = f"{args.base_url.rstrip('/')}/v1/ocr/{kind}/batch"
    source_card_ids: tuple[str, ...] = ()
    if kind == "id-card":
        archive, source_card_ids = id_archive(args.id_cards, count)
        files: Any = {"archive": ("cards.zip", archive, "application/zip")}
    else:
        image = args.passport if kind == "passport" else args.driving_license
        mime = "image/png" if image.suffix.lower() == ".png" else "image/jpeg"
        files = [("images", (f"{index:03d}-{image.name}", image.read_bytes(), mime)) for index in range(count)]
    started = time.perf_counter()
    try:
        response = requests.post(url, files=files, timeout=args.timeout)
    except requests.RequestException as error:
        return Measurement(args.runtime, kind, count, repeat, time.perf_counter() - started, None, "request_failed", str(error), {}, [], [], None, None, None, None, {})
    client_total = time.perf_counter() - started
    if not response.ok:
        return Measurement(args.runtime, kind, count, repeat, client_total, None, "request_failed", response.text[:1000], {}, [], [], None, None, None, None, {})
    try:
        payload = response.json()
        diagnostics = payload["diagnostics"]
        server_total = float(payload["total_seconds"])
        if int(payload["succeeded"]) + int(payload["failed"]) != count:
            raise ValueError(f"response count differs from N={count}")
        localization = tensor_batches(diagnostics.get("localization"))
        detection = tensor_batches(diagnostics.get("text_detection")).get("default", [])
        recognition = tensor_batches(diagnostics.get("text_recognition")).get("default", [])
        line_filter = diagnostics.get("line_filter", {})
        recognition_seconds = float(diagnostics.get("text_recognition", {}).get("wall_seconds", 0.0))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return Measurement(args.runtime, kind, count, repeat, client_total, None, "invalid_response", str(error), {}, [], [], None, None, None, None, {})
    return Measurement(
        args.runtime, kind, count, repeat, client_total, server_total, "ok", None,
        localization, detection, recognition,
        int(line_filter.get("detected_line_count", 0)),
        int(line_filter.get("recognition_candidate_count", 0)),
        int(line_filter.get("filtered_before_recognition_count", 0)),
        recognition_seconds / server_total if server_total else None,
        stage_seconds(diagnostics, server_total),
        tensor_batches(diagnostics.get("mrz_recognition")).get("default", []),
        int(diagnostics.get("id_card_mrz_probe", {}).get("front_fallback_scanned", 0)),
        diagnostics.get("text_recognition", {}).get("packing_strategy"),
        int(payload["succeeded"]),
        int(payload["failed"]),
        source_card_ids,
    )


def proves_batching(row: Measurement) -> bool:
    sizes = [size for values in row.localization_tensor_batches.values() for size in values]
    return row.document_count > 1 and max(sizes + row.detection_tensor_batches + row.recognition_tensor_batches, default=0) > 1


def summarize(rows: list[Measurement], kinds: tuple[str, ...]) -> list[dict[str, Any]]:
    summary = []
    for kind in kinds:
        for count in sorted({row.document_count for row in rows if row.document_type == kind}):
            values = [row for row in rows if row.document_type == kind and row.document_count == count and row.status == "ok"]
            if not values:
                continue
            summary.append({
                "document_type": kind,
                "document_count": count,
                "requests": len(values),
                "total_client_seconds": statistics.median(row.client_total_seconds for row in values),
                "total_server_seconds": statistics.median(row.server_total_seconds or 0.0 for row in values),
                "batching_proven": any(proves_batching(row) for row in values),
                "localization_tensor_batches": values[0].localization_tensor_batches,
                "detection_tensor_batches": values[0].detection_tensor_batches,
                "recognition_tensor_batches": values[0].recognition_tensor_batches,
                "mrz_recognition_tensor_batches": values[0].mrz_recognition_tensor_batches,
                "id_card_front_fallback_scan_count": values[0].id_card_front_fallback_scan_count,
                "recognition_packing_strategy": values[0].recognition_packing_strategy,
                "succeeded": sum(row.succeeded or 0 for row in values),
                "failed": sum(row.failed or 0 for row in values),
                "source_card_ids": values[0].source_card_ids,
                "total_stage_seconds": {stage: statistics.median(row.stage_seconds.get(stage, 0.0) for row in values) for stage in values[0].stage_seconds},
            })
    return summary


def plot(path: Path, summary: list[dict[str, Any]], runtime: str, kinds: tuple[str, ...]) -> None:
    figure = plt.figure(figsize=(10, 6))
    for kind in kinds:
        rows = [row for row in summary if row["document_type"] == kind]
        plt.scatter([row["document_count"] for row in rows], [row["total_server_seconds"] for row in rows], label=kind)
    plt.xlabel("Logical documents in one request")
    plt.ylabel("Median server batch time (seconds)")
    plt.title(f"Voight true-batch median time ({runtime})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def plot_stages(path: Path, summary: list[dict[str, Any]], runtime: str) -> None:
    stages = tuple(next((row["total_stage_seconds"] for row in summary), {}))
    bottom = [0.0] * len(summary)
    figure = plt.figure(figsize=(13, 7))
    labels = [f"{row['document_type']} N={row['document_count']}" for row in summary]
    for stage in stages:
        values = [row["total_stage_seconds"][stage] for row in summary]
        plt.bar(labels, values, bottom=bottom, label=stage)
        bottom = [current + value for current, value in zip(bottom, values)]
    plt.ylabel("Median server seconds")
    plt.title(f"Voight batch stage time ({runtime})")
    plt.xticks(rotation=30, ha="right")
    plt.legend(fontsize="small", ncol=2)
    plt.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def main() -> int:
    args = parse_args()
    try:
        response = requests.get(f"{args.base_url.rstrip('/')}/v1/health/ready", timeout=min(args.timeout, 30))
        response.raise_for_status()
    except requests.RequestException as error:
        print(f"API is not ready: {error}", file=sys.stderr)
        return 2
    if not args.no_warmup:
        for kind in args.kinds:
            warmup = post(args, kind, 1, 0)
            if warmup.status != "ok":
                print(f"Warm-up {kind} failed: {warmup.failure_detail}", file=sys.stderr)
                return 2
    rows = [post(args, kind, count, repeat) for kind in args.kinds for count in args.sizes for repeat in range(1, args.repeats + 1)]
    summary = summarize(rows, args.kinds)
    output = args.output_dir or ROOT / "complexity" / "results" / f"{args.runtime}-{datetime.now():%Y%m%dT%H%M%S}"
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "runtime": args.runtime,
        "base_url": args.base_url,
        "document_type": args.document_type,
        "max_batch_size": args.max_batch_size,
        "sizes": args.sizes,
        "repeats": args.repeats,
        "summary_statistic": "median",
        "id_card_source_dir": str(args.id_card_dir) if args.id_card_dir else None,
        "id_card_source_cards": [card_id for card_id, _, _ in args.id_cards],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (output / "batch_measurements.json").write_text(json.dumps({"run": metadata, "rows": [asdict(row) for row in rows], "summary": summary}, indent=2), encoding="utf-8")
    with (output / "batch_measurements.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(asdict(rows[0])) if rows else list(Measurement.__annotations__))
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    plot(output / "batch_total_time.png", summary, args.runtime, args.kinds)
    plot_stages(output / "batch_stage_time.png", summary, args.runtime)
    for row in summary:
        print(f"{row['document_type']:16} N={row['document_count']:2} total={row['total_server_seconds']:.3f}s batching={row['batching_proven']}")
    return 1 if any(row.status != "ok" for row in rows) or any(row["document_count"] > 1 and not row["batching_proven"] for row in summary) else 0


if __name__ == "__main__":
    raise SystemExit(main())
