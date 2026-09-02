"""Benchmark recognition crop packing on the corrected CPU pipeline."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.maintained.model_matrix_benchmark import Server, _available_memory, _post, _score
from benchmarks.maintained.batch_size_benchmark import _differences, _snapshot
from benchmarks.maintained.pipeline_breakdown import validate_and_manifest

DOC_TYPES = ("passport", "id_card", "driving_license")
STRATEGIES = ("current", "aspect-ratio", "fixed-width-buckets", "best-fit")
PACKING_ENV = {"current": "fixed-width", "aspect-ratio": "aspect-ratio"}
BASE_ENV = {
    "RUNTIME_TARGET": "cpu",
    "OCR_DEVICE": "cpu",
    "CPU_THREADS": "4",
    "TEXT_RECOGNITION_PROCESSES": "1",
    "LOCALIZATION_BATCH_SIZE": "4",
    "TEXT_DETECTION_BATCH_SIZE": "1",
    "TEXT_RECOGNITION_BATCH_SIZE": "2",
    "MRZ_RECOGNITION_BATCH_SIZE": "2",
    "TEXT_DETECTOR_MODEL": "PP-OCRv6_medium_det",
    "TEXT_RECOGNIZER_MODEL": "latin_PP-OCRv5_mobile_rec",
    "DOCALIGNER_MODEL": "fastvit_sa24",
    "DOCALIGNER_MODEL_TYPE": "heatmap",
    "MRZ_RECOGNIZER_BACKEND": "generic-paddle",
    "MRZ_RECOGNIZER_MODEL": "20250221",
    "OCR_MAX_SIDE": "3000",
    "OCR_CONTRAST": "1.25",
    "MRZ_POLYGON_PADDING_RATIO": "0.03",
    "DOCALIGNER_PADDING": "100",
    "DRIVING_LICENSE_MIN_OVERLAP_RATIO": "0.30",
}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/benchmark")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/benchmarks/12.recognition-packing-comparison")
    parser.add_argument("--port", type=int, default=8017)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--recognition-batch-size", type=int, default=2)
    parser.add_argument("--strategies", nargs="+", choices=STRATEGIES, default=list(STRATEGIES))
    args = parser.parse_args()
    if args.repeats < 3 or args.warmup != 1 or args.recognition_batch_size <= 0:
        parser.error("use one warm-up, at least three measured repeats, and a positive recognition batch size")
    return args


def _env(strategy: str, batch_size: int) -> dict[str, str]:
    return {
        **BASE_ENV,
        "TEXT_RECOGNITION_BATCH_SIZE": str(batch_size),
        "TEXT_RECOGNITION_PACKING": PACKING_ENV.get(strategy, strategy),
    }


def _verify_loaded(ready: dict[str, Any], env: dict[str, str]) -> None:
    models = ready.get("models", {})
    expected = {
        "text_detector": {"model": env["TEXT_DETECTOR_MODEL"]},
        "text_recognizer": {"model": env["TEXT_RECOGNIZER_MODEL"]},
        "document_localizer": {"model_cfg": env["DOCALIGNER_MODEL"], "model_type": env["DOCALIGNER_MODEL_TYPE"]},
        "mrz_localizer": {"model_cfg": "20250222"},
        "mrz_recognizer": {"backend": env["MRZ_RECOGNIZER_BACKEND"]},
    }
    for section, checks in expected.items():
        actual = models.get(section, {})
        if not actual.get("loaded"):
            raise RuntimeError(f"{section} did not load: {actual}")
        for key, value in checks.items():
            if actual.get(key) != value:
                raise RuntimeError(f"{section}.{key}: expected {value!r}, got {actual.get(key)!r}")


def _records(payload: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for call in payload.get("diagnostics", {}).get("text_recognition", {}).get("calls", []):
        result.extend(call.get("crop_records", []))
    return result


def _padding(records: list[dict[str, Any]]) -> dict[str, Any]:
    useful = sum(int(row.get("useful_pixels", 0)) for row in records)
    padded = sum(int(row.get("padded_pixels", 0)) for row in records)
    batches = Counter(row.get("batch_index") for row in records)
    batch_sizes = Counter(size for size in batches.values())
    tensor_shapes = Counter((row.get("tensor_h"), row.get("tensor_w")) for row in records)
    return {
        "crop_count": len(records),
        "useful_pixels": useful,
        "padded_pixels": padded,
        "padding_efficiency": useful / padded if padded else None,
        "batch_count": len(batches),
        "batch_size_distribution": dict(sorted(batch_sizes.items())),
        "tensor_shape_distribution": {f"{h}x{w}": count for (h, w), count in sorted(tensor_shapes.items())},
    }


def _crop_invariant(before: list[dict[str, Any]], after: list[dict[str, Any]]) -> dict[str, Any]:
    key = lambda row: (row.get("sample_id"), row.get("line_index"))
    left = {key(row): (row.get("crop_sha256"), row.get("original_crop_h"), row.get("original_crop_w")) for row in before}
    right = {key(row): (row.get("crop_sha256"), row.get("original_crop_h"), row.get("original_crop_w")) for row in after}
    mismatches = [identifier for identifier in sorted(set(left) | set(right)) if left.get(identifier) != right.get(identifier)]
    return {"identical": not mismatches and left.keys() == right.keys(), "compared": len(left), "mismatches": mismatches}


def _row(strategy: str, kind: str, repeat: int, payload: dict[str, Any], client_seconds: float, score: dict[str, Any], lifecycle: dict[str, Any], current: dict[str, Any] | None) -> dict[str, Any]:
    diagnostics = payload.get("diagnostics", {})
    records = _records(payload)
    useful = sum(int(row.get("useful_pixels", 0)) for row in records)
    padded = sum(int(row.get("padded_pixels", 0)) for row in records)
    snapshot = _snapshot([], payload)
    difference = _differences([], current or {}, snapshot)
    confidence_differences = 0
    if current is not None:
        current_items = {item.get("input", {}).get("document_id"): item for item in current.get("items", [])}
        for item in payload.get("items", []):
            identifier = item.get("input", {}).get("document_id")
            left = current_items.get(identifier, {}).get("result") or {}
            right = item.get("result") or {}
            if (left.get("fields") or {}) != (right.get("fields") or {}):
                confidence_differences += 1
    return {
        "strategy": strategy,
        "document_type": kind,
        "repeat": repeat,
        "status": "ok" if payload.get("failed", 0) == 0 else "partial",
        "e2e_seconds": float(payload.get("total_seconds", client_seconds)),
        "client_e2e_seconds": client_seconds,
        "docs_per_second": len(payload.get("items", [])) / client_seconds if client_seconds else None,
        "recognition_seconds": float(diagnostics.get("text_recognition", {}).get("elapsed_wall_seconds", 0.0)),
        "peak_rss_mb": max(float(diagnostics.get("process_peak_rss_mb", 0.0)), float(lifecycle.get("peak_process_memory_mb", 0.0))),
        "recognition_padding_efficiency": useful / padded if padded else None,
        "useful_pixels": useful,
        "padded_pixels": padded,
        "recognition_crop_count": len(records),
        "tensor_batch_count": len({row.get("batch_index") for row in records}),
        "tensor_batch_sizes": dict(Counter(Counter(row.get("batch_index") for row in records).values())),
        "tensor_shapes": dict(Counter(f"{row.get('tensor_h')}x{row.get('tensor_w')}" for row in records)),
        "correctness": score.get("field_correctness"),
        "mrz_correctness": score.get("mrz_exact_match_rate"),
        "mrz_line_accuracy": score.get("mrz_line_accuracy"),
        "failures": payload.get("failed", 0),
        "output_differences": len(difference),
        "confidence_only_differences": confidence_differences,
        "crop_content_identical": None if current is None else _crop_invariant(_records(current), records)["identical"],
    }


def main() -> int:
    args = arguments()
    documents, manifest = validate_and_manifest(args.dataset_root)
    by_kind = {kind: [doc for doc in documents if doc.document_type == kind] for kind in DOC_TYPES}
    output = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    (output / "server_logs").mkdir()
    configs = [{"strategy": strategy, "packing": _env(strategy, args.recognition_batch_size)["TEXT_RECOGNITION_PACKING"], "env": _env(strategy, args.recognition_batch_size)} for strategy in args.strategies]
    (output / "manifest.json").write_text(json.dumps({"dataset": manifest, "configs": configs, "repeats": args.repeats, "warmup": args.warmup, "fresh_server_per_config": True, "cpu_only": True, "thread_configuration_source": "outputs/benchmarks/11.cpu-thread-count-benchmark/comparison.csv; 4 threads was fastest mean E2E"}, indent=2), encoding="utf-8")

    rows: list[dict[str, Any]] = []
    responses: dict[str, dict[str, dict[int, dict[str, Any]]]] = defaultdict(lambda: defaultdict(dict))
    for number, config in enumerate(configs, 1):
        strategy = config["strategy"]
        run_dir = output / f"{number:02d}.{strategy}-packing"
        run_dir.mkdir()
        server = Server(args, run_dir, config["env"])
        lifecycle: dict[str, Any] = {"memory_before_mb": (_available_memory() or 0) / 1024 / 1024}
        try:
            ready = server.start()
            _verify_loaded(ready, config["env"])
            (run_dir / "loaded_configuration.json").write_text(json.dumps(ready, indent=2), encoding="utf-8")
            for kind, docs in by_kind.items():
                payload, seconds = _post(kind, docs, args.port, args.timeout)
                (run_dir / "warmup").mkdir(exist_ok=True)
                (run_dir / "warmup" / f"{kind}.json").write_text(json.dumps({"client_seconds": seconds, "response": payload}, indent=2), encoding="utf-8")
            for repeat in range(1, args.repeats + 1):
                for kind, docs in by_kind.items():
                    payload, seconds = _post(kind, docs, args.port, args.timeout)
                    responses[strategy][kind][repeat] = payload
                    raw_path = run_dir / "raw" / f"{repeat:02d}.repeat-{repeat}"
                    raw_path.mkdir(parents=True, exist_ok=True)
                    (raw_path / f"{kind}.json").write_text(json.dumps({"client_seconds": seconds, "response": payload}, indent=2), encoding="utf-8")
                    current_payload = responses["current"][kind].get(repeat) if strategy != "current" else None
                    rows.append(_row(strategy, kind, repeat, payload, seconds, _score(kind, docs, payload), lifecycle, current_payload))
                    server._sample()
        except Exception as error:
            (run_dir / "failure.json").write_text(json.dumps({"error": f"{type(error).__name__}: {error}"}, indent=2), encoding="utf-8")
            raise
        finally:
            lifecycle.update(server.stop())
            available = _available_memory()
            lifecycle["memory_after_shutdown_mb"] = available / 1024 / 1024 if available else None
            (run_dir / "lifecycle.json").write_text(json.dumps(lifecycle, indent=2), encoding="utf-8")
        if not lifecycle.get("cleanup_verified"):
            raise RuntimeError(f"server cleanup failed for {strategy}: {lifecycle}")
        if strategy == "current":
            observed = {
                kind: dict(Counter(row.get("natural_resized_w") for payload in responses[strategy][kind].values() for row in _records(payload)))
                for kind in DOC_TYPES
            }
            (output / "observed_crop_width_distribution.json").write_text(json.dumps({"source": "current strategy measured crops", "widths": observed}, indent=2), encoding="utf-8")

    # Re-score differences after the current reference exists for all repeats.
    for row in rows:
        if row["strategy"] == "current":
            continue
        baseline = responses["current"][row["document_type"]].get(row["repeat"])
        candidate = responses[row["strategy"]][row["document_type"]][row["repeat"]]
        row["output_differences"] = len(_differences(documents=[doc for doc in by_kind[row["document_type"]]], before=_snapshot(by_kind[row["document_type"]], baseline), after=_snapshot(by_kind[row["document_type"]], candidate)))

    with (output / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = sorted({field for row in rows for field in row})
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows(rows)
    (output / "aggregate.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")

    output_differences = []
    for strategy in args.strategies:
        if strategy == "current":
            continue
        for kind in DOC_TYPES:
            for repeat, candidate in responses[strategy][kind].items():
                baseline = responses["current"][kind][repeat]
                output_differences.extend({"strategy": strategy, "document_type": kind, "repeat": repeat, **difference} for difference in _differences(by_kind[kind], _snapshot(by_kind[kind], baseline), _snapshot(by_kind[kind], candidate)))
    (output / "output_differences.json").write_text(json.dumps(output_differences, indent=2, ensure_ascii=False), encoding="utf-8")

    crop_rows = []
    for strategy, kinds in responses.items():
        for kind, repeats in kinds.items():
            for repeat, payload in repeats.items():
                for record in _records(payload):
                    crop_rows.append({"strategy": strategy, "document_type": kind, "repeat": repeat, **record})
    with (output / "recognition_crops.jsonl").open("w", encoding="utf-8") as handle:
        handle.write("\n".join(json.dumps(row, ensure_ascii=False) for row in crop_rows) + "\n")

    summary = []
    for strategy in args.strategies:
        for kind in DOC_TYPES:
            values = [row for row in rows if row["strategy"] == strategy and row["document_type"] == kind]
            if not values:
                continue
            numeric = ("e2e_seconds", "client_e2e_seconds", "docs_per_second", "recognition_seconds", "recognition_padding_efficiency", "useful_pixels", "padded_pixels", "peak_rss_mb", "tensor_batch_count")
            summary.append({"strategy": strategy, "document_type": kind, "repeats": len(values), **{f"median_{key}": statistics.median(available) if (available := [row[key] for row in values if row[key] is not None]) else None for key in numeric}, "tensor_batch_sizes": json.dumps(dict(Counter(size for row in values for size, count in row["tensor_batch_sizes"].items() for _ in range(count))), sort_keys=True), "tensor_shapes": json.dumps(dict(Counter(shape for row in values for shape, count in row["tensor_shapes"].items() for _ in range(count))), sort_keys=True), "correctness": values[0]["correctness"], "mrz_correctness": values[0]["mrz_correctness"], "output_differences": sum(row["output_differences"] for row in values), "failures": sum(row["failures"] for row in values), "crop_content_identical": all(row["crop_content_identical"] is not False for row in values)})
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({field for row in summary for field in row})); writer.writeheader(); writer.writerows(summary)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
