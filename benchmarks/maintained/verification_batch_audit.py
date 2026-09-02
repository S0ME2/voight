"""Summarize one current verification benchmark into a batch audit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.inference.packing import FixedWidthBatchPacker

STAGES = (
    "localization",
    "mrz_localization",
    "text_detection",
    "text_recognition",
    "mrz_recognition",
)
LIMITS = {
    "localization": ("LOCALIZATION_BATCH_SIZE", 4),
    "mrz_localization": ("LOCALIZATION_BATCH_SIZE", 4),
    "text_detection": ("TEXT_DETECTION_BATCH_SIZE", 1),
    "text_recognition": ("TEXT_RECOGNITION_BATCH_SIZE", 2),
    "mrz_recognition": ("MRZ_RECOGNITION_BATCH_SIZE", 2),
}
DOC_TYPES = ("passport", "id_card", "driving_license", "overall")


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    return statistics.quantiles(values, n=100, method="inclusive")[int(fraction * 100) - 1]


def stats(values: list[float]) -> dict[str, float | None]:
    return {
        "min": min(values) if values else None,
        "median": statistics.median(values) if values else None,
        "mean": statistics.mean(values) if values else None,
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "max": max(values) if values else None,
        "stddev": statistics.stdev(values) if len(values) > 1 else 0.0 if values else None,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, separators=(",", ":")) if isinstance(value, (list, dict)) else value for key, value in row.items()})


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _detail(row: dict[str, Any], stage: str) -> dict[str, Any]:
    return row.get("actual_tensor_batches", {}).get(stage, {})


def _input_count(row: dict[str, Any], stage: str) -> int:
    if stage == "text_detection":
        return sum(_detail(row, stage).get("submitted_batch_sizes", []))
    if stage == "text_recognition":
        return int(row.get("recognition_candidate_count", 0))
    return 0


def _batches(row: dict[str, Any], stage: str) -> list[int]:
    return list(_detail(row, stage).get("tensor_batch_sizes", []))


def _all_requests(raw: list[dict[str, Any]], phase: str, repeat: int | None = None) -> list[dict[str, Any]]:
    return [row for row in raw if row.get("phase") == phase and (repeat is None or row.get("repeat") == repeat)]


def _doc_workloads(ocr_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in sorted(ocr_rows, key=lambda value: value["document_id"]):
        rows.append({
            "document_type": row["document_type"],
            "document_id": row["document_id"],
            "localization_input_count": "n/a",
            "localization_tensor_batches": "n/a",
            "mrz_localization_input_count": "n/a",
            "mrz_localization_tensor_batches": "n/a",
            "detection_input_count": _input_count(row, "text_detection"),
            "detection_tensor_batches": _batches(row, "text_detection"),
            "recognition_candidate_count": _input_count(row, "text_recognition"),
            "recognition_tensor_batches": _batches(row, "text_recognition"),
            "mrz_recognition_input_count": "n/a",
            "mrz_recognition_tensor_batches": "n/a",
        })
    return rows


def _summary_rows(workloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for stage, key in (("text_detection", "detection_input_count"), ("text_recognition", "recognition_candidate_count")):
        for document_type in DOC_TYPES:
            selected = [row for row in workloads if document_type == "overall" or row["document_type"] == document_type]
            values = [float(row[key]) for row in selected]
            rows.append({"stage": stage, "document_type": document_type, "applicable": True, "document_count": len(selected), **stats(values)})
    for stage in ("localization", "mrz_localization", "mrz_recognition"):
        rows.append({"stage": stage, "document_type": "overall", "applicable": False, "document_count": len(workloads), "min": "n/a", "median": "n/a", "mean": "n/a", "p90": "n/a", "p95": "n/a", "max": "n/a", "stddev": "n/a"})
    return rows


def _cap_rows(workloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for stage, key in (("text_detection", "detection_input_count"), ("text_recognition", "recognition_candidate_count")):
        env_name, limit = LIMITS[stage]
        values = [int(row[key]) for row in workloads]
        above = sum(value > limit for value in values)
        exact = sum(value == limit for value in values)
        below = sum(value < limit for value in values)
        rows.append({"stage": stage, "setting": env_name, "current_limit": limit, "documents_below_limit": below, "documents_exactly_at_limit": exact, "documents_above_limit": above, "cap_binding_rate": above / len(values) if values else None, "applicable": True})
    for stage in ("localization", "mrz_localization", "mrz_recognition"):
        env_name, limit = LIMITS[stage]
        rows.append({"stage": stage, "setting": env_name, "current_limit": limit, "documents_below_limit": "n/a", "documents_exactly_at_limit": "n/a", "documents_above_limit": "n/a", "cap_binding_rate": "n/a", "applicable": False})
    return rows


def _recognition_bucket_counts(row: dict[str, Any]) -> list[int]:
    counts: dict[int, int] = {}
    packer = FixedWidthBatchPacker()
    for call in _detail(row, "text_recognition").get("calls", []):
        heights = call.get("input_heights", [])
        widths = call.get("input_widths", [])
        for height, width in zip(heights, widths):
            image_shape = type("ImageShape", (), {"shape": (height, width)})()
            bucket = packer._bucket(image_shape)
            counts[bucket] = counts.get(bucket, 0) + 1
    return list(counts.values())


def _detection_shape_counts(row: dict[str, Any]) -> list[int]:
    counts: dict[tuple[int, int], int] = {}
    for call in _detail(row, "text_detection").get("calls", []):
        for shape in call.get("tensor_shapes", []):
            key = (int(shape[-2]), int(shape[-1]))
            counts[key] = counts.get(key, 0) + int(shape[0])
    return list(counts.values())


def _hypothetical_rows(ocr_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for stage, limits, group_fn in (
        ("text_detection", (1, 2, 4), _detection_shape_counts),
        ("text_recognition", (1, 2, 4, 8, 12, 16, 24, 32, 64), _recognition_bucket_counts),
    ):
        groups_by_doc = {row["document_id"]: group_fn(row) for row in ocr_rows}
        total_inputs = sum(_input_count(row, stage) for row in ocr_rows)
        for limit in limits:
            calls_by_doc = {doc: sum(math.ceil(count / limit) for count in groups) for doc, groups in groups_by_doc.items()}
            split = sum(calls > 1 for calls in calls_by_doc.values())
            values = list(calls_by_doc.values())
            rows.append({"stage": stage, "hypothetical_limit": limit, "documents": len(values), "total_real_inputs": total_inputs, "total_inference_calls": sum(values), "mean_calls_per_document": statistics.mean(values) if values else None, "max_calls_per_document": max(values) if values else None, "documents_split": split, "split_rate": split / len(values) if values else None})
    return rows


def _doc_timing_rows(raw: list[dict[str, Any]], documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ocr = {(row["repeat"], row["document_id"]): row for row in _all_requests(raw, "ocr")}
    checks: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in _all_requests(raw, "check"):
        checks.setdefault((row["repeat"], row["document_id"]), []).append(row)
    rows = []
    for repeat, document_id in sorted(ocr):
        row = ocr[(repeat, document_id)]
        document_type = row["document_type"]
        check_rows = checks.get((repeat, document_id), [])
        check_seconds = sum(float(check.get("server_latency_seconds") or 0.0) for check in check_rows)
        ocr_stages = row.get("stages", {})
        check_stages = [check.get("stages", {}) for check in check_rows]
        rows.append({
            "repeat": repeat,
            "document_type": document_type,
            "document_id": document_id,
            "localization_seconds": "n/a",
            "mrz_localization_seconds": "n/a",
            "canonicalization_seconds": "n/a",
            "text_detection_seconds": float(ocr_stages.get("text_detection", 0.0)),
            "text_recognition_seconds": float(ocr_stages.get("text_recognition", 0.0)),
            "mrz_recognition_seconds": "n/a",
            "verification_seconds": check_seconds,
            "result_assembly_seconds": float(ocr_stages.get("response_construction", 0.0)) + sum(float(stage.get("response_construction", 0.0)) for stage in check_stages),
            "end_to_end_seconds": float(row.get("server_latency_seconds") or 0.0) + check_seconds,
        })
    return rows


def _timing_summary(timing_rows: list[dict[str, Any]], raw: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("localization_seconds", "mrz_localization_seconds", "canonicalization_seconds", "text_detection_seconds", "text_recognition_seconds", "mrz_recognition_seconds", "verification_seconds", "result_assembly_seconds", "end_to_end_seconds")
    all_ocr = _all_requests(raw, "ocr")
    model_calls = {stage: sum(int(_detail(row, stage).get("model_call_count", len(_batches(row, stage)))) for row in all_ocr) for stage in ("text_detection", "text_recognition")}
    real_inputs = {stage: sum(_input_count(row, stage) for row in all_ocr) for stage in ("text_detection", "text_recognition")}
    rows = []
    for key in keys:
        values = [float(row[key]) for row in timing_rows if row[key] != "n/a"]
        model_stage = key in {"text_detection_seconds", "text_recognition_seconds"}
        stage = key.removesuffix("_seconds")
        calls = model_calls.get(stage, 0)
        inputs = real_inputs.get(stage, 0)
        rows.append({"stage": stage, "scope": "overall", "documents": len(values), "unit": "seconds/document", **stats(values), "total_seconds": sum(values), "model_calls": calls if model_stage else "n/a", "real_inputs": inputs if model_stage else "n/a", "seconds_per_call": sum(values) / calls if model_stage and calls else "n/a", "seconds_per_real_input": sum(values) / inputs if model_stage and inputs else "n/a"})
    return rows


def _call_summary(ocr_rows: list[dict[str, Any]], timing: list[dict[str, Any]], repeat_count: int) -> list[dict[str, Any]]:
    rows = []
    timing_by_stage = {row["stage"]: row for row in timing}
    for stage, key in (("text_detection", "detection_input_count"), ("text_recognition", "recognition_candidate_count")):
        calls = [int(_detail(row, stage).get("model_call_count", len(_batches(row, stage)))) for row in ocr_rows]
        inputs = [int(_input_count(row, stage)) for row in ocr_rows]
        total_calls, total_inputs = sum(calls), sum(inputs)
        total_seconds = float(timing_by_stage[stage]["total_seconds"])
        rows.append({"stage": stage, "scope": "one 20-document pass", "documents": len(ocr_rows), "total_real_inputs": total_inputs, "total_model_calls": total_calls, "mean_calls_per_document": statistics.mean(calls), "seconds_per_call": total_seconds / repeat_count / total_calls if total_calls else None, "seconds_per_real_input": total_seconds / repeat_count / total_inputs if total_inputs else None, **stats([float(value) for value in calls])})
    for stage in ("localization", "mrz_localization", "mrz_recognition"):
        rows.append({"stage": stage, "scope": "overall", "documents": len(ocr_rows), "total_real_inputs": "n/a", "total_model_calls": "n/a", "mean_calls_per_document": "n/a", "seconds_per_call": "n/a", "seconds_per_real_input": "n/a", "min": "n/a", "median": "n/a", "mean": "n/a", "p90": "n/a", "p95": "n/a", "max": "n/a", "stddev": "n/a"})
    return rows


def _accuracy(raw_dir: Path) -> dict[str, Any]:
    rows = []
    with (raw_dir / "field_results.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    first = [row for row in rows if row.get("repeat") == "1"]
    accepted = sum(row.get("status") in {"match", "likely_match"} for row in first)
    counts = {name: sum(row.get("status") == name for row in first) for name in ("match", "likely_match", "mismatch", "not_found")}
    return {"unique_fields": len(first), "accepted": accepted, "accuracy": accepted / len(first) if first else None, "hard_failures": counts["mismatch"] + counts["not_found"], **counts}


def _report(path: Path, config: dict[str, Any], workloads: list[dict[str, Any]], summaries: list[dict[str, Any]], caps: list[dict[str, Any]], hypothetical: list[dict[str, Any]], timing: list[dict[str, Any]], calls: list[dict[str, Any]], accuracy: dict[str, Any], lifecycle: list[dict[str, Any]]) -> None:
    settings = config["settings"]
    overall_detection = next(row for row in summaries if row["stage"] == "text_detection" and row["document_type"] == "overall")
    overall_recognition = next(row for row in summaries if row["stage"] == "text_recognition" and row["document_type"] == "overall")
    cap_by_stage = {row["stage"]: row for row in caps}
    timing_by_stage = {row["stage"]: row for row in timing}
    call_by_stage = {row["stage"]: row for row in calls}
    lines = [
        "# Verification Batch-Size Audit", "", "## Decision", "",
        "No production batch-size change is justified by this audit alone. The current verification route uses detection and recognition; recognition is binding by input count, but its width-bucket packing creates many real calls and the existing configuration must be evaluated against that observed workload before tuning.",
        "", "## Current settings", "",
        f"- Localization: `{settings['localization']['current_value']}` (`LOCALIZATION_BATCH_SIZE`; also used for MRZ localization in `ProfileBatchRunner`).",
        f"- MRZ localization: `{settings['mrz_localization']['current_value']}`; no separate current setting.",
        f"- Detection: `{settings['text_detection']['current_value']}` (`TEXT_DETECTION_BATCH_SIZE`).",
        f"- Recognition: `{settings['text_recognition']['current_value']}` (`TEXT_RECOGNITION_BATCH_SIZE`).",
        f"- MRZ recognition: `{settings['mrz_recognition']['current_value']}` (`MRZ_RECOGNITION_BATCH_SIZE`).",
        "- Sources/consumers: `app/config.py:248-251` (`Settings.from_env`), `app/models.py:143-167` (`profile_batch_runner`), and `app/inference/batch.py:331-355`, `app/inference/batch.py:491-527` (`BatchedOcr`).",
        "", "## Actual workload", "",
        f"- Overall detection inputs/document: `{overall_detection['min']}/{overall_detection['median']}/{overall_detection['mean']:.2f}/{overall_detection['p90']}/{overall_detection['p95']}/{overall_detection['max']}` (min/median/mean/p90/p95/max).",
        f"- Overall recognition candidates/document: `{overall_recognition['min']}/{overall_recognition['median']}/{overall_recognition['mean']:.2f}/{overall_recognition['p90']}/{overall_recognition['p95']}/{overall_recognition['max']}`.",
        "- Localization, MRZ localization, and specialized MRZ recognition are not called by the current whole-image `/verification/*/ocr` route; their workload values are `n/a`, not zero.",
        "- Per-document evidence is in `document_workloads.csv`; distributions by document type are in `workload_summary.csv`.",
        "", "## Current cap-binding", "",
        f"- Detection: `{cap_by_stage['text_detection']['documents_above_limit']}/{len(workloads)}` = `{cap_by_stage['text_detection']['cap_binding_rate']:.0%}`.",
        f"- Recognition: `{cap_by_stage['text_recognition']['documents_above_limit']}/{len(workloads)}` = `{cap_by_stage['text_recognition']['cap_binding_rate']:.0%}`.",
        "- The three non-route stages are not applicable.",
        "", "## Inference calls", "",
        f"- Detection: `{call_by_stage['text_detection']['total_model_calls']}` calls across `{len(workloads)}` documents; median `{call_by_stage['text_detection']['median']}` calls/document.",
        f"- Recognition: `{call_by_stage['text_recognition']['total_model_calls']}` calls across `{len(workloads)}` documents; median `{call_by_stage['text_recognition']['median']}` calls/document.",
        "- Full hypothetical call counts for limits 1–64 are in `hypothetical_partitioning.csv`; they are calculated from observed width/shape groups, not timed.",
        "", "## Stage timing", "",
    ]
    for stage in ("text_detection", "text_recognition", "verification", "result_assembly", "end_to_end"):
        row = timing_by_stage[stage]
        lines.append(f"- `{stage}`: median `{row['median']:.4f}s/document`, mean `{row['mean']:.4f}s`, p95 `{row['p95']:.4f}s`, stddev `{row['stddev']:.4f}s`.")
    lines += [
        "- Current benchmark instrumentation does not invoke localization, MRZ localization, canonicalization, or specialized MRZ recognition for this route; those rows are `n/a`.",
        "- `stage_timings.csv` includes total seconds, model-call references, and per-real-input/per-call calculations where available.",
        "", "## Accuracy sanity check", "",
        f"- `{accuracy['unique_fields']}` unique fields; accepted `{accuracy['accepted']}`; accuracy `{accuracy['accuracy']:.2%}`; hard failures `{accuracy['hard_failures']}` (`{accuracy['mismatch']} mismatch + {accuracy['not_found']} not_found`).",
        "", "## Batch semantics proof", "",
        "- DocAligner/MRZ localizer heatmap adapters concatenate real inputs into a dynamic-N tensor (`app/inference/localization.py:45-67`, `:76-97`); point adapters explicitly execute one image at a time (`:104-121`). The runner partitions jobs with `app/inference/batch.py:795-799`.",
        "- Paddle detection is caller-limited, then shape-buckets and spatially pads images inside `_fixed_shape_predict` (`app/inference/paddle.py:45-109`). Batch dimension is the real group length; spatial dimensions may be padded.",
        "- Paddle recognition receives the already prepared crop list and calls `model.predict(input=list(images), batch_size=len(images))` (`app/inference/paddle.py:188-205`). `FixedWidthBatchPacker.pack` (`app/inference/packing.py:74-96`) groups by width and applies the configured maximum. A 15-item same-width group is `[15]`, not padded to 32.",
        "- MRZScanner recognition concatenates real dynamic-N tensors when supported and otherwise explicitly loops batch-one (`app/inference/mrzscanner.py:24-73`).",
        "", "## Memory", "",
        f"- Baseline process RSS after model load: median `{statistics.median(row['server_rss_baseline_mb'] for row in lifecycle):.1f} MB`; peak RSS: median `{statistics.median(row['peak_process_memory_mb'] for row in lifecycle):.1f} MB`, maximum `{max(row['peak_process_memory_mb'] for row in lifecycle):.1f} MB`; `{sum(bool(row.get('cleanup_verified')) for row in lifecycle)}/{len(lifecycle)}` processes cleaned up successfully.",
        "- Configured maxima are passed to the caller/runner, not allocated as fixed model memory. Runtime tensor memory follows actual grouped inputs; detection additionally allocates spatially padded tensors for each shape group.",
        "", "## Route/model sharing", "",
        "- Verification uses the same cached text detector and recognizer returned by `Models.text_detector()`/`Models.text_recognizer()` (`app/models.py:189-199`). Its OCR coordinator is separate, but models/sessions are shared.",
        "- Current verification route does not use DocAligner or MRZScanner. The profile runner uses the cached localization/MRZ adapters (`app/models.py:159-167`). Different caller partitioning can share loaded model instances where the adapter supports it.",
        "", "## Files", "",
        "- Per-document workloads: `document_workloads.csv`.",
        "- Distributions/caps/call projections: `workload_summary.csv`, `batch_cap_analysis.csv`, `hypothetical_partitioning.csv`, `model_call_summary.csv`.",
        "- Timing/environment: `stage_timings.csv`, `environment.json`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    raw = load_jsonl(args.benchmark_dir / "raw_runs.jsonl")
    ocr_rows = _all_requests(raw, "ocr", 1)
    if len(ocr_rows) != 20:
        raise SystemExit(f"expected 20 01.repeat-1 OCR rows, got {len(ocr_rows)}")
    workloads = _doc_workloads(ocr_rows)
    summaries = _summary_rows(workloads)
    caps = _cap_rows(workloads)
    hypothetical = _hypothetical_rows(ocr_rows)
    timing_rows = _doc_timing_rows(raw, workloads)
    timing = _timing_summary(timing_rows, raw)
    calls = _call_summary(ocr_rows, timing, 5)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = json.loads((args.benchmark_dir / "environment.json").read_text(encoding="utf-8"))
    environment["audit"] = {"source_benchmark": str(args.benchmark_dir), "measured_repeats": 5, "warmup_passes": 1, "http_batching": False, "route": "current whole-image verification OCR/check workload"}
    (args.output_dir / "environment.json").write_text(json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8")
    config = {
        "settings": {
            "localization": {"setting_name": "LOCALIZATION_BATCH_SIZE", "current_value": 4, "env_value": 4, "env_example_value": 4, "python_default": 4, "python_field": "Settings.runtime.localization_batch_size", "consumed_by": "Models.profile_batch_runner -> ProfileBatchRunner.localization_batch_size", "source": ".env; app/config.py:248; app/config.py:78"},
            "mrz_localization": {"setting_name": "LOCALIZATION_BATCH_SIZE", "current_value": 4, "env_value": 4, "env_example_value": 4, "python_default": 4, "python_field": "Settings.runtime.localization_batch_size", "consumed_by": "ProfileBatchRunner._localize for the MRZ localizer; no separate MRZ-localization setting", "source": ".env; app/config.py:248; app/inference/batch.py:795-799"},
            "text_detection": {"setting_name": "TEXT_DETECTION_BATCH_SIZE", "current_value": 1, "env_value": 1, "env_example_value": 1, "python_default": 1, "python_field": "Settings.runtime.text_detection_batch_size", "consumed_by": "Models.profile_batch_runner and Models.verification_ocr -> BatchedOcr.detection_batch_size", "source": ".env; app/config.py:249; app/inference/batch.py:507-523"},
            "text_recognition": {"setting_name": "TEXT_RECOGNITION_BATCH_SIZE", "current_value": 2, "env_value": 2, "env_example_value": 2, "python_default": 2, "python_field": "Settings.runtime.text_recognition_batch_size", "consumed_by": "Models.profile_batch_runner and Models.verification_ocr -> BatchedOcr.recognition_batch_size", "source": ".env; app/config.py:250; app/inference/batch.py:617-627"},
            "mrz_recognition": {"setting_name": "MRZ_RECOGNITION_BATCH_SIZE", "current_value": 2, "env_value": 2, "env_example_value": 2, "python_default": 2, "python_field": "Settings.runtime.mrz_recognition_batch_size", "consumed_by": "ProfileBatchRunner.mrz_recognition_batch_size; current verification route does not invoke specialized MRZ recognition", "source": ".env; app/config.py:251; app/inference/batch.py:1122"},
        },
        "consumers": {"profile_runner": "Models.profile_batch_runner -> ProfileBatchRunner", "verification_route": "Models.verification_ocr -> BatchedOcr", "model_instances": "Models.text_detector(), Models.text_recognizer(), Models.document_localizer(), Models.mrz_localizer(), and Models.mrz_recognizer() are cached per Models owner"},
        "semantics": "caller/runtime maximums; actual tensor batch dimension follows real grouped inputs",
        "verification_route_applicability": {"localization": False, "mrz_localization": False, "text_detection": True, "text_recognition": True, "mrz_recognition": False},
    }
    (args.output_dir / "current_batch_config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    write_csv(args.output_dir / "document_workloads.csv", workloads)
    write_csv(args.output_dir / "workload_summary.csv", summaries)
    write_csv(args.output_dir / "batch_cap_analysis.csv", caps)
    write_csv(args.output_dir / "hypothetical_partitioning.csv", hypothetical)
    write_csv(args.output_dir / "stage_timings.csv", timing)
    write_csv(args.output_dir / "model_call_summary.csv", calls)
    accuracy = _accuracy(args.benchmark_dir)
    lifecycle = []
    for path in sorted(args.benchmark_dir.glob("[0-9][0-9].repeat-*/lifecycle.json")):
        lifecycle.append(json.loads(path.read_text(encoding="utf-8")))
    environment["memory"] = {
        "baseline_process_rss_after_model_load_mb": stats([float(row["server_rss_baseline_mb"]) for row in lifecycle]),
        "peak_process_rss_mb": stats([float(row["peak_process_memory_mb"]) for row in lifecycle]),
        "all_processes_cleanup_verified": all(bool(row.get("cleanup_verified")) for row in lifecycle),
    }
    (args.output_dir / "environment.json").write_text(json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8")
    _report(args.output_dir / "report.md", config, workloads, summaries, caps, hypothetical, timing, calls, accuracy, lifecycle)
    (args.output_dir / "accuracy_summary.json").write_text(json.dumps(accuracy, indent=2), encoding="utf-8")
    print(f"completed verification batch audit: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
