"""Benchmark one pipeline batch size at a time with a fresh CPU server."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.maintained.model_matrix_benchmark import (  # reuse the tested lifecycle
    Server,
    _available_memory,
    _post,
    _score,
    _stage_totals,
)
from benchmarks.maintained.pipeline_breakdown import annotation_truth, discover_dataset, validate_and_manifest

DOC_TYPES = ("passport", "id_card", "driving_license")
STAGES = ("localization", "text_detection", "text_recognition")
BATCH_SIZES = (1, 2, 4, 8, 16, 32)
BASELINE = {
    "LOCALIZATION_BATCH_SIZE": "16",
    "TEXT_DETECTION_BATCH_SIZE": "16",
    "TEXT_RECOGNITION_BATCH_SIZE": "32",
    "MRZ_RECOGNITION_BATCH_SIZE": "16",
}
MODEL_ENV = {
    "TEXT_DETECTOR_MODEL": "PP-OCRv6_medium_det",
    "TEXT_RECOGNIZER_MODEL": "latin_PP-OCRv5_mobile_rec",
    "DOCALIGNER_MODEL": "fastvit_sa24",
    "DOCALIGNER_MODEL_TYPE": "heatmap",
    "MRZ_RECOGNIZER_BACKEND": "generic-paddle",
    "MRZ_RECOGNIZER_MODEL": "20250221",
    "TEXT_RECOGNITION_PACKING": "fixed-width",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", ".paddlex")))
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/benchmarks/09.batch-size-sweep")
    parser.add_argument("--port", type=int, default=8012)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=("text_detection", "text_recognition"))
    args = parser.parse_args()
    if args.repeats < 3 or args.warmup != 1:
        parser.error("this procedure requires exactly one warm-up and at least three measured repeats")
    return args


def _configuration(stage: str, size: int) -> dict[str, str]:
    values = dict(BASELINE)
    values[{"localization": "LOCALIZATION_BATCH_SIZE", "text_detection": "TEXT_DETECTION_BATCH_SIZE", "text_recognition": "TEXT_RECOGNITION_BATCH_SIZE"}[stage]] = str(size)
    return {**MODEL_ENV, **values}


def _verify_loaded(ready: dict[str, Any], env: dict[str, str]) -> None:
    models = ready.get("models", {})
    checks = {
        "text_detector": {"model": env["TEXT_DETECTOR_MODEL"]},
        "text_recognizer": {"model": env["TEXT_RECOGNIZER_MODEL"]},
        "document_localizer": {"model_cfg": env["DOCALIGNER_MODEL"], "model_type": env["DOCALIGNER_MODEL_TYPE"]},
        "mrz_localizer": {"model_cfg": "20250222"},
        "mrz_recognizer": {
            "backend": env["MRZ_RECOGNIZER_BACKEND"],
            "model_cfg": env["MRZ_RECOGNIZER_MODEL"],
            "model_cfg_used": False,
            "effective_backend": "paddle",
            "effective_model": env["TEXT_RECOGNIZER_MODEL"],
            "source": "text_recognizer",
        },
    }
    for section, expected in checks.items():
        actual = models.get(section, {})
        if not actual.get("loaded"):
            raise RuntimeError(f"{section} is not loaded: {actual}")
        for key, value in expected.items():
            if actual.get(key) != value:
                raise RuntimeError(f"{section}.{key}: expected {value!r}, got {actual.get(key)!r}")


def _configured_sizes(payload: dict[str, Any]) -> dict[str, Any]:
    diagnostics = payload.get("diagnostics", {})
    localization = {}
    for name, stage in diagnostics.get("localization", {}).items():
        if isinstance(stage, dict) and "configured_batch_size" in stage:
            localization[name] = stage["configured_batch_size"]
    return {
        "localization": localization,
        "text_detection": diagnostics.get("text_detection", {}).get("configured_batch_size"),
        "text_recognition": diagnostics.get("text_recognition", {}).get("configured_batch_size"),
        "mrz_recognition": diagnostics.get("mrz_recognition", {}).get("configured_batch_size"),
    }


def _stage_detail(payload: dict[str, Any], kind: str) -> dict[str, Any]:
    diagnostics = payload.get("diagnostics", {})
    stage = diagnostics.get(kind, {})
    calls = stage.get("calls", []) if isinstance(stage, dict) else []
    tensor_sizes = [size for call in calls for size in call.get("tensor_batch_sizes", [call.get("tensor_batch_size")]) if size is not None]
    submitted = [int(call["submitted_batch_size"]) for call in calls if "submitted_batch_size" in call]
    efficiencies = []
    for call in calls:
        value = call.get("padding_efficiency")
        if value is None:
            value = call.get("recognition_width_padding_efficiency")
        if value is not None:
            efficiencies.append(float(value))
    return {
        "configured_batch_size": stage.get("configured_batch_size"),
        "tensor_batch_sizes": tensor_sizes,
        "submitted_batch_sizes": submitted,
        "tensor_batch_count": len(tensor_sizes),
        "item_count": sum(submitted),
        "padding_efficiency_mean": statistics.mean(efficiencies) if efficiencies else None,
        "padding_efficiency_min": min(efficiencies) if efficiencies else None,
        "model_call_count": stage.get("model_call_count", len(tensor_sizes)),
        "failure_count": stage.get("failure_count", 0),
        "elapsed_wall_seconds": stage.get("elapsed_wall_seconds", stage.get("wall_seconds", 0.0)),
        "calls": calls,
    }


def _localization_detail(payload: dict[str, Any]) -> dict[str, Any]:
    localization = payload.get("diagnostics", {}).get("localization", {})
    details = {}
    for name, stage in localization.items():
        if isinstance(stage, dict):
            details[name] = {
                "configured_batch_size": stage.get("configured_batch_size"),
                "tensor_batch_sizes": [size for call in stage.get("calls", []) for size in call.get("tensor_batch_sizes", [call.get("tensor_batch_size")]) if size is not None],
                "submitted_batch_sizes": [call.get("submitted_batch_size") for call in stage.get("calls", [])],
                "tensor_batch_count": len(stage.get("tensor_batch_sizes", [])),
                "item_count": sum(call.get("submitted_batch_size", 0) for call in stage.get("calls", [])),
                "failure_count": stage.get("failure_count", 0),
                "elapsed_wall_seconds": stage.get("wall_seconds", 0.0),
                "calls": stage.get("calls", []),
            }
    return details


def _snapshot(documents: list[Any], payload: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for document, item in zip(documents, payload.get("items", [])):
        value = item.get("result") or {}
        result[document.document_id] = {
            "success": item.get("success", False),
            "error": item.get("error"),
            "fields": {name: entry.get("value") for name, entry in (value.get("fields") or {}).items()},
            "mrz": (value.get("mrz") or {}).get("raw_lines", []),
        }
    return result


def _differences(documents: list[Any], before: dict[str, Any], after: dict[str, Any]) -> list[dict[str, Any]]:
    differences = []
    for document in documents:
        identifier = document.document_id
        left, right = before.get(identifier, {}), after.get(identifier, {})
        fields = sorted(set(left.get("fields", {})) | set(right.get("fields", {})))
        changed_fields = [{"field": name, "baseline": left.get("fields", {}).get(name), "batch": right.get("fields", {}).get(name)} for name in fields if left.get("fields", {}).get(name) != right.get("fields", {}).get(name)]
        if left.get("success") != right.get("success") or left.get("mrz", []) != right.get("mrz", []) or changed_fields:
            differences.append({"document_id": identifier, "changed_fields": changed_fields, "baseline_mrz": left.get("mrz", []), "batch_mrz": right.get("mrz", []), "baseline_success": left.get("success"), "batch_success": right.get("success"), "baseline_error": left.get("error"), "batch_error": right.get("error")})
    return differences


def _row(config: dict[str, Any], kind: str, repeat: int, payload: dict[str, Any], client_seconds: float, score: dict[str, Any], lifecycle: dict[str, Any]) -> dict[str, Any]:
    diagnostics = payload.get("diagnostics", {})
    details = {name: _stage_detail(payload, name) for name in ("text_detection", "text_recognition", "mrz_recognition")}
    details["localization"] = _localization_detail(payload)
    stages = _stage_totals([payload])
    total_items = len(payload.get("items", []))
    row = {
        "configuration": config["name"], "changed_stage": config["stage"], "requested_batch_size": config["size"],
        "document_type": kind, "repeat": repeat, "status": "ok" if payload.get("failed", 0) == 0 else "partial",
        "logical_items": total_items, "physical_images": sum(d["physical_image_count"] for d in config["dataset"]["documents"] if d["document_type"] == kind),
        "total_latency_seconds": float(payload.get("total_seconds", client_seconds)), "client_latency_seconds": client_seconds,
        "throughput_logical_docs_per_second": total_items / client_seconds if client_seconds else None,
        "peak_rss_mb": max([float(diagnostics.get("process_peak_rss_mb", 0.0)), float(lifecycle.get("peak_process_memory_mb", 0.0))]),
        "failed_items": payload.get("failed", 0), "errors": [item.get("error") for item in payload.get("items", []) if not item.get("success")],
        "field_correctness": score["field_correctness"], "field_exact": score["field_exact"], "field_total": score["field_total"],
        "visible_character_accuracy": score.get("field_character_accuracy"), "visible_characters_correct": score.get("field_characters"), "visible_character_total": score.get("field_character_total"), "visible_character_errors": (score.get("field_character_total", 0) - score.get("field_characters", 0)), "document_correctness": score["document_correctness"],
        "mrz_exact_match_rate": score["mrz_exact_match_rate"], "mrz_line_accuracy": score["mrz_line_accuracy"], "mrz_character_accuracy": score["mrz_character_accuracy"],
        "mrz_characters_correct": score["mrz_characters"], "mrz_character_total": score["mrz_character_total"], "mrz_character_errors": score["mrz_character_total"] - score["mrz_characters"], "mrz_full_exact": score["mrz_full_exact"], "mrz_documents": score["mrz_documents"], "stages": stages, "batch_details": details,
        "configured_batch_sizes": _configured_sizes(payload),
        "recognition_crop_count": diagnostics.get("line_filter", {}).get("recognition_candidate_count", 0),
        "detected_line_count": diagnostics.get("line_filter", {}).get("detected_line_count", 0),
    }
    for name, value in stages.items():
        row[f"{name}_seconds"] = value
    for name, detail in details.items():
        if name == "localization":
            continue
        row[f"{name}_tensor_batch_sizes"] = detail["tensor_batch_sizes"]
        row[f"{name}_tensor_batch_count"] = detail["tensor_batch_count"]
        row[f"{name}_item_count"] = detail["item_count"]
        row[f"{name}_padding_efficiency_mean"] = detail["padding_efficiency_mean"]
        row[f"{name}_padding_efficiency_min"] = detail["padding_efficiency_min"]
    return row


def main() -> int:
    args = _args()
    documents, manifest = validate_and_manifest(args.dataset_root)
    by_kind = {kind: [doc for doc in documents if doc.document_type == kind] for kind in DOC_TYPES}
    output = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    (output / "server_logs").mkdir()
    all_configs = [{"name": f"{stage}_batch_{size}", "stage": stage, "size": size, "env": _configuration(stage, size)} for stage in args.stages for size in BATCH_SIZES]
    (output / "manifest.json").write_text(json.dumps({"dataset": manifest, "baseline": BASELINE, "models": MODEL_ENV, "configs": all_configs, "repeats": args.repeats, "warmup": args.warmup, "fresh_server_per_config": True, "cpu_only": True, "procedure": "one full-corpus warm-up and three full-corpus measured repeats per document type"}, indent=2), encoding="utf-8")
    rows, snapshots, failures, route_evidence = [], {}, [], []
    for index, config in enumerate(all_configs, 1):
        print(f"[{index}/{len(all_configs)}] {config['name']}", flush=True)
        config_dir = output / f"{index:02d}.{config['name'].replace('_', '-') }"
        config_dir.mkdir()
        env = config["env"]
        server = Server(args, config_dir, env)
        lifecycle = {"memory_before_mb": (_available_memory() or 0) / 1024 / 1024}
        measured_snapshots = {}
        try:
            ready = server.start()
            _verify_loaded(ready, env)
            route_evidence.append(ready.get("models", {}).get("mrz_recognizer", {}))
            (config_dir / "loaded_configuration.json").write_text(json.dumps(ready, indent=2), encoding="utf-8")
            warmup = {}
            for kind, docs in by_kind.items():
                payload, seconds = _post(kind, docs, args.port, args.timeout)
                warmup[kind] = {"client_wall_seconds": seconds, "response": payload}
                (config_dir / "warmup").mkdir(exist_ok=True)
                (config_dir / "warmup" / f"{kind}.json").write_text(json.dumps(warmup[kind], indent=2), encoding="utf-8")
            loaded_sizes = {kind: _configured_sizes(data["response"]) for kind, data in warmup.items()}
            expected_sizes = {"localization": {"docaligner": int(env["LOCALIZATION_BATCH_SIZE"]), "mrz": int(env["LOCALIZATION_BATCH_SIZE"])}, "text_detection": int(env["TEXT_DETECTION_BATCH_SIZE"]), "text_recognition": int(env["TEXT_RECOGNITION_BATCH_SIZE"]), "mrz_recognition": int(env["MRZ_RECOGNITION_BATCH_SIZE"])}
            if any(loaded.get("text_detection") != expected_sizes["text_detection"] or loaded.get("text_recognition") != expected_sizes["text_recognition"] or loaded.get("mrz_recognition") != expected_sizes["mrz_recognition"] or any(value != expected_sizes["localization"].get(name) for name, value in loaded.get("localization", {}).items()) for loaded in loaded_sizes.values()):
                raise RuntimeError(f"configured batch size verification failed: expected={expected_sizes} actual={loaded_sizes}")
            (config_dir / "batch_size_verification.json").write_text(json.dumps({"expected": expected_sizes, "observed_in_warmup": loaded_sizes}, indent=2), encoding="utf-8")
            for repeat in range(1, args.repeats + 1):
                repeat_dir = config_dir / "raw" / f"{repeat:02d}.repeat-{repeat}"
                repeat_dir.mkdir(parents=True)
                for kind, docs in by_kind.items():
                    payload, seconds = _post(kind, docs, args.port, args.timeout)
                    score = _score(kind, docs, payload)
                    row = _row(config | {"dataset": manifest}, kind, repeat, payload, seconds, score, lifecycle)
                    rows.append(row)
                    measured_snapshots.setdefault(kind, {})[repeat] = _snapshot(docs, payload)
                    (repeat_dir / f"{kind}.json").write_text(json.dumps({"configuration": config, "client_wall_seconds": seconds, "score": score, "response": payload}, indent=2), encoding="utf-8")
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
    raw_path = output / "raw_measurements.jsonl"
    raw_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    fields = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))})
    with (output / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows({key: row.get(key) for key in fields} for row in rows)
    aggregate = []
    for config in all_configs:
        for kind in DOC_TYPES:
            values = [row for row in rows if row["configuration"] == config["name"] and row["document_type"] == kind]
            if not values:
                continue
            numeric = ["total_latency_seconds", "client_latency_seconds", "throughput_logical_docs_per_second", "peak_rss_mb", "localization_seconds", "text_detection_seconds", "text_recognition_seconds", "mrz_recognition_seconds"]
            aggregate.append({"configuration": config["name"], "changed_stage": config["stage"], "requested_batch_size": config["size"], "document_type": kind, "repeats": len(values), **{f"median_{key}": statistics.median(row[key] for row in values) for key in numeric}, "field_correctness": values[0]["field_correctness"], "visible_character_accuracy": values[0]["visible_character_accuracy"], "visible_character_errors": values[0]["visible_character_errors"], "mrz_exact_match_rate": values[0]["mrz_exact_match_rate"], "mrz_line_accuracy": values[0]["mrz_line_accuracy"], "mrz_character_errors": values[0]["mrz_character_errors"], "peak_rss_mb": max(row["peak_rss_mb"] for row in values), "actual_tensor_batches": json.dumps({name: values[0].get(f"{name}_tensor_batch_sizes") for name in ("text_detection", "text_recognition")}, separators=(",", ":")), "padding_efficiency": json.dumps({name: values[0].get(f"{name}_padding_efficiency_mean") for name in ("text_detection", "text_recognition")}, separators=(",", ":"))})
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted({key for row in aggregate for key in row})); writer.writeheader(); writer.writerows(aggregate)
    (output / "summary.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    baseline_snapshots = snapshots.get("text_recognition_batch_32", {})
    differences = []
    for config_name, config_snapshots in snapshots.items():
        for kind, repeats in config_snapshots.items():
            if config_name == "text_recognition_batch_32":
                continue
            for repeat, snapshot in repeats.items():
                differences.extend({"configuration": config_name, "document_type": kind, "repeat": repeat, **difference} for difference in _differences(by_kind[kind], baseline_snapshots.get(kind, {}).get(repeat, {}), snapshot))
    (output / "output_differences.json").write_text(json.dumps(differences, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    (output / "model_route.json").write_text(json.dumps({"expected": {"backend": "generic-paddle", "configured_model": "20250221", "configured_model_used": False, "effective_backend": "paddle", "effective_model": MODEL_ENV["TEXT_RECOGNIZER_MODEL"], "source": "text_recognizer"}, "observed": route_evidence}, indent=2), encoding="utf-8")
    print(f"completed {len(rows)} measured rows; output: {output}", flush=True)
    return 2 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
