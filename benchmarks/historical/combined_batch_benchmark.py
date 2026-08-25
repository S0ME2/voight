"""Benchmark selected end-to-end batch combinations with fresh servers."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.maintained.batch_size_benchmark import (
    BATCH_SIZES,
    DOC_TYPES,
    MODEL_ENV,
    _configuration,
    _differences,
    _post,
    _row,
    _snapshot,
    _verify_loaded,
    _configured_sizes,
)
from benchmarks.maintained.model_matrix_benchmark import Server, _available_memory
from benchmarks.maintained.pipeline_breakdown import validate_and_manifest


CONFIGS = ((4, 8, 2), (4, 16, 2), (4, 8, 4), (1, 8, 2))


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", ".paddlex")))
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/benchmarks/batch_size/CORRECTED_COMBINED_RUN")
    parser.add_argument("--baseline-root", type=Path, default=ROOT / "outputs/benchmarks/batch_size/CORRECTED_RUN")
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if args.repeats < 3 or args.warmup != 1:
        parser.error("this procedure requires exactly one warm-up and at least three measured repeats")
    return args


def _env(localization: int, detection: int, recognition: int) -> dict[str, str]:
    return {
        **MODEL_ENV,
        "LOCALIZATION_BATCH_SIZE": str(localization),
        "TEXT_DETECTION_BATCH_SIZE": str(detection),
        "TEXT_RECOGNITION_BATCH_SIZE": str(recognition),
        "MRZ_RECOGNITION_BATCH_SIZE": "16",
    }


def _config(localization: int, detection: int, recognition: int) -> dict[str, object]:
    return {
        "name": f"localization_{localization}_detection_{detection}_recognition_{recognition}",
        "stage": "combined",
        "size": {"localization": localization, "text_detection": detection, "text_recognition": recognition},
        "env": _env(localization, detection, recognition),
    }


def _baseline_snapshots(root: Path, documents: dict[str, list[object]]) -> dict[str, dict[int, dict[str, object]]]:
    base = root if (root / "raw").is_dir() else root / "text_recognition_batch_32"
    snapshots = {}
    for kind, docs in documents.items():
        snapshots[kind] = {}
        for repeat in (1, 2, 3):
            path = base / "raw" / f"repeat-{repeat}" / f"{kind}.json"
            if path.is_file():
                snapshots[kind][repeat] = _snapshot(docs, json.loads(path.read_text())['response'])
    return snapshots


def main() -> int:
    args = _args()
    documents, manifest = validate_and_manifest(args.dataset_root)
    by_kind = {kind: [doc for doc in documents if doc.document_type == kind] for kind in DOC_TYPES}
    if not (args.baseline_root / "raw").is_dir() and not (args.baseline_root / "text_recognition_batch_32").is_dir():
        raise RuntimeError(f"corrected baseline is missing: {args.baseline_root}")
    output = args.output_root
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    (output / "server_logs").mkdir()
    configs = [_config(*values) for values in CONFIGS]
    (output / "manifest.json").write_text(json.dumps({"dataset": manifest, "configs": configs, "repeats": args.repeats, "warmup": args.warmup, "fresh_server_per_config": True, "cpu_only": True, "baseline": str(args.baseline_root / "text_recognition_batch_32")}, indent=2), encoding="utf-8")
    rows, snapshots, failures, routes = [], {}, [], []
    for index, config in enumerate(configs, 1):
        print(f"[{index}/{len(configs)}] {config['name']}", flush=True)
        config_dir = output / str(config["name"])
        config_dir.mkdir()
        env = config["env"]
        server = Server(args, config_dir, env)
        lifecycle = {"memory_before_mb": (_available_memory() or 0) / 1024 / 1024}
        measured_snapshots = {}
        try:
            ready = server.start()
            _verify_loaded(ready, env)
            routes.append(ready.get("models", {}).get("mrz_recognizer", {}))
            (config_dir / "loaded_configuration.json").write_text(json.dumps(ready, indent=2), encoding="utf-8")
            warmup_dir = config_dir / "warmup"
            warmup_dir.mkdir()
            warmup_sizes = {}
            for kind, docs in by_kind.items():
                payload, seconds = _post(kind, docs, args.port, args.timeout)
                warmup_sizes[kind] = _configured_sizes(payload)
                (warmup_dir / f"{kind}.json").write_text(json.dumps({"client_wall_seconds": seconds, "response": payload}, indent=2), encoding="utf-8")
            expected = {"localization": {"docaligner": int(config["size"]["localization"]), "mrz": int(config["size"]["localization"])}, "text_detection": int(config["size"]["text_detection"]), "text_recognition": int(config["size"]["text_recognition"]), "mrz_recognition": 16}
            if any(
                observed.get("text_detection") != expected["text_detection"]
                or observed.get("text_recognition") != expected["text_recognition"]
                or observed.get("mrz_recognition") != expected["mrz_recognition"]
                or any(value != expected["localization"].get(name) for name, value in observed.get("localization", {}).items())
                for observed in warmup_sizes.values()
            ):
                raise RuntimeError(f"configured batch size verification failed: expected={expected} actual={warmup_sizes}")
            (config_dir / "batch_size_verification.json").write_text(json.dumps({"expected": expected, "observed_in_warmup": warmup_sizes}, indent=2), encoding="utf-8")
            for repeat in range(1, args.repeats + 1):
                repeat_dir = config_dir / "raw" / f"repeat-{repeat}"
                repeat_dir.mkdir(parents=True)
                for kind, docs in by_kind.items():
                    payload, seconds = _post(kind, docs, args.port, args.timeout)
                    from benchmarks.maintained.model_matrix_benchmark import _score
                    row = _row(config | {"dataset": manifest}, kind, repeat, payload, seconds, _score(kind, docs, payload), lifecycle)
                    rows.append(row)
                    measured_snapshots.setdefault(kind, {})[repeat] = _snapshot(docs, payload)
                    (repeat_dir / f"{kind}.json").write_text(json.dumps({"configuration": config, "client_wall_seconds": seconds, "response": payload}, indent=2), encoding="utf-8")
                server._sample()
            snapshots[config["name"]] = measured_snapshots
        except Exception as error:
            failure = {"configuration": config, "error": f"{type(error).__name__}: {error}"}
            failures.append(failure)
            (config_dir / "failure.json").write_text(json.dumps(failure, indent=2), encoding="utf-8")
            print(f"    FAILED: {failure['error']}", flush=True)
        finally:
            cleanup = server.stop()
            memory_after = _available_memory()
            lifecycle.update(cleanup, memory_after_shutdown_mb=memory_after / 1024 / 1024 if memory_after else None)
            (config_dir / "lifecycle.json").write_text(json.dumps(lifecycle, indent=2), encoding="utf-8")
        if not lifecycle.get("cleanup_verified", False):
            raise RuntimeError(f"server cleanup failed for {config['name']}: {lifecycle}")

    (output / "raw_measurements.jsonl").write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    fields = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))})
    with (output / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows({key: row.get(key) for key in fields} for row in rows)
    summary = []
    numeric = ("total_latency_seconds", "client_latency_seconds", "throughput_logical_docs_per_second", "peak_rss_mb", "localization_seconds", "text_detection_seconds", "text_recognition_seconds", "mrz_recognition_seconds")
    for config in configs:
        for kind in DOC_TYPES:
            values = [row for row in rows if row["configuration"] == config["name"] and row["document_type"] == kind]
            if not values:
                continue
            summary.append({"configuration": config["name"], "document_type": kind, "repeats": len(values), **{f"median_{key}": statistics.median(row[key] for row in values) for key in numeric}, "field_correctness": values[0]["field_correctness"], "visible_character_accuracy": values[0]["visible_character_accuracy"], "visible_character_errors": values[0]["visible_character_errors"], "mrz_exact_match_rate": values[0]["mrz_exact_match_rate"], "mrz_line_accuracy": values[0]["mrz_line_accuracy"], "mrz_character_errors": values[0]["mrz_character_errors"], "failures": sum(row["failed_items"] for row in values), "peak_rss_mb": max(row["peak_rss_mb"] for row in values), "actual_tensor_batches": json.dumps({name: values[0].get(f"{name}_tensor_batch_sizes") for name in ("text_detection", "text_recognition")}, separators=(",", ":")), "padding_efficiency": json.dumps({name: values[0].get(f"{name}_padding_efficiency_mean") for name in ("text_detection", "text_recognition")}, separators=(",", ":"))})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in summary for key in row})); writer.writeheader(); writer.writerows(summary)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    baseline = _baseline_snapshots(args.baseline_root, by_kind)
    differences = []
    for name, config_snapshots in snapshots.items():
        for kind, repeats in config_snapshots.items():
            for repeat, snapshot in repeats.items():
                differences.extend({"configuration": name, "document_type": kind, "repeat": repeat, **difference} for difference in _differences(by_kind[kind], baseline.get(kind, {}).get(repeat, {}), snapshot))
    (output / "output_differences.json").write_text(json.dumps(differences, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    (output / "model_route.json").write_text(json.dumps({"expected": {"backend": "generic-paddle", "configured_model": "20250221", "configured_model_used": False, "effective_backend": "paddle", "effective_model": MODEL_ENV["TEXT_RECOGNIZER_MODEL"], "source": "text_recognizer"}, "observed": routes}, indent=2), encoding="utf-8")
    print(f"completed {len(rows)} measured rows; output: {output}", flush=True)
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
