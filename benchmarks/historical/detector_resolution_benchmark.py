"""Measure detector resolution while keeping canonical recognition crops full size."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.maintained.model_matrix_benchmark import Server, _post
from benchmarks.maintained.pipeline_breakdown import DOC_TYPES, annotation_truth, validate_and_manifest

SCALES = (100.0, 87.5, 75.0, 62.5, 50.0)
BASE_ENV = {
    "RUNTIME_TARGET": "cpu", "OCR_DEVICE": "cpu", "CPU_THREADS": "4",
    "LOCALIZATION_BATCH_SIZE": "16", "TEXT_DETECTION_BATCH_SIZE": "16",
    "TEXT_RECOGNITION_BATCH_SIZE": "32", "MRZ_RECOGNITION_BATCH_SIZE": "16",
    "TEXT_RECOGNITION_PROCESSES": "1", "TEXT_RECOGNITION_PACKING": "aspect-ratio",
    "TEXT_DETECTOR_MODEL": "PP-OCRv6_medium_det",
    "TEXT_RECOGNIZER_MODEL": "latin_PP-OCRv5_mobile_rec",
    "DOCALIGNER_MODEL": "fastvit_sa24", "DOCALIGNER_MODEL_TYPE": "heatmap",
    "MRZ_RECOGNIZER_BACKEND": "generic-paddle", "MRZ_RECOGNIZER_MODEL": "20250221",
}


def aligned_limit(scale: float, current: int = 960) -> int:
    return max(32, int(scale / 100 * current / 32 + 0.5) * 32)


def _distance(left: str, right: str) -> int:
    row = list(range(len(right) + 1))
    for i, char in enumerate(left, 1):
        next_row = [i]
        for j, other in enumerate(right, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (char != other)))
        row = next_row
    return row[-1]


def _signature(item: dict[str, Any]) -> dict[str, Any]:
    result = item.get("result") or {}
    return {
        "success": item.get("success"),
        "fields": {name: value.get("value") for name, value in (result.get("fields") or {}).items()},
        "mrz": (result.get("mrz") or {}).get("raw_lines", []),
    }


def _correctness(documents: list[Any], payload: dict[str, Any]) -> dict[str, Any]:
    fields = exact = chars = char_total = missing = extra = 0
    mrz_docs = mrz_found = mrz_exact = mrz_lines = mrz_line_exact = mrz_chars = mrz_char_total = 0
    for document, item in zip(documents, payload.get("items", [])):
        truth = annotation_truth(document)
        actual = _signature(item)["fields"]
        expected_names = set()
        for name, entry in truth.get("fields", {}).items():
            if not isinstance(entry, dict) or entry.get("state") not in {"value", "empty"}:
                continue
            expected_names.add(name)
            expected = entry.get("value") if entry.get("state") == "value" else None
            value = actual.get(name)
            expected_text, actual_text = "" if expected is None else str(expected), "" if value is None else str(value)
            fields += 1
            chars += max(len(expected_text), 1) - _distance(expected_text, actual_text)
            char_total += max(len(expected_text), 1)
            if expected is not None and value in (None, ""):
                missing += 1
            if expected is None and value not in (None, ""):
                extra += 1
            exact += value == expected or (expected is None and value in (None, ""))
        extra += sum(value not in (None, "") for name, value in actual.items() if name not in expected_names)
        expected_lines = [line for line in truth.get("mrz", {}).get("lines", []) if isinstance(line, str)]
        if expected_lines:
            actual_lines = _signature(item)["mrz"]
            mrz_docs += 1
            mrz_found += bool(actual_lines)
            mrz_exact += actual_lines == expected_lines
            for index, expected in enumerate(expected_lines):
                value = actual_lines[index] if index < len(actual_lines) else ""
                mrz_lines += 1
                mrz_line_exact += value == expected
                mrz_chars += len(expected) - _distance(expected, value)
                mrz_char_total += len(expected)
    return {
        "visible_exact": exact, "visible_total": fields,
        "visible_character_accuracy": chars / char_total if char_total else None,
        "missing_fields": missing, "extra_fields": extra,
        "mrz_found": mrz_found, "mrz_documents": mrz_docs, "mrz_exact": mrz_exact,
        "mrz_line_exact": mrz_line_exact, "mrz_lines": mrz_lines,
        "mrz_character_accuracy": mrz_chars / mrz_char_total if mrz_char_total else None,
    }


def _measurement(kind: str, documents: list[Any], payload: dict[str, Any], client_seconds: float, repeat: int, scale: float, limit: int, baseline: dict[str, Any] | None) -> dict[str, Any]:
    diagnostics = payload.get("diagnostics", {})
    detection = diagnostics.get("text_detection", {})
    line_filter = diagnostics.get("line_filter", {})
    calls = detection.get("calls", [])
    tensor_shapes = [shape for call in calls for shape in call.get("tensor_shapes", [])]
    tensor_pixels = [value for call in calls for value in call.get("tensor_pixel_counts", [])]
    crop_records = [record for call in diagnostics.get("text_recognition", {}).get("calls", []) for record in call.get("crop_records", [])]
    crop_signature = [(record.get("sample_id"), record.get("line_index"), record.get("crop_sha256"), record.get("original_crop_h"), record.get("original_crop_w")) for record in crop_records]
    outputs = [_signature(item) for item in payload.get("items", [])]
    differences = []
    if baseline is not None:
        for index, (before, after) in enumerate(zip(baseline["outputs"], outputs)):
            if before != after:
                differences.append({"document_id": documents[index].document_id, "baseline": before, "current": after})
    score = _correctness(documents, payload)
    return {
        "scale_percent": scale, "detector_limit_side_len": limit, "document_type": kind, "repeat": repeat,
        "status": "ok" if payload.get("failed", 0) == 0 else "failed",
        "client_e2e_seconds": client_seconds, "e2e_seconds": payload.get("total_seconds"),
        "docs_per_second": len(documents) / payload["total_seconds"] if payload.get("total_seconds") else None,
        "detection_seconds": detection.get("elapsed_wall_seconds", detection.get("wall_seconds")),
        "detector_tensor_shapes": tensor_shapes, "detector_tensor_pixel_counts": tensor_pixels,
        "detector_tensor_pixel_count_total": sum(tensor_pixels),
        "detected_line_count": line_filter.get("detected_line_count", 0),
        "recognition_candidate_crop_count": line_filter.get("recognition_candidate_count", 0),
        "detected_line_count_by_sample": line_filter.get("samples", {}),
        "recognition_crop_signature": crop_signature,
        "recognition_crop_content_identical_vs_100": None if baseline is None else crop_signature == baseline.get("recognition_crop_signature"),
        "peak_rss_mb": diagnostics.get("process_peak_rss_mb"),
        "failures": payload.get("failed", 0), "output_difference_count_vs_100": len(differences),
        "output_differences_vs_100": differences, "outputs": outputs, **score,
    }


def _median(rows: list[dict[str, Any]], key: str) -> Any:
    values = [row[key] for row in rows if row.get(key) is not None]
    return statistics.median(values) if values else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/benchmark")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/benchmarks/13.detector-resolution-workload")
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        raise SystemExit("--repeats must be at least 3")
    documents, manifest = validate_and_manifest(args.dataset_root)
    output = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=True)
    limits = {str(scale): aligned_limit(scale) for scale in SCALES}
    (output / "manifest.json").write_text(json.dumps({
        "dataset": manifest, "models": BASE_ENV, "scales_percent": SCALES,
        "detector_current_config_source": "PaddleOCR TextDetection predictor Resize transform",
        "target_limit_side_len": limits, "warmup": 1, "measured_repeats": args.repeats,
        "fresh_server_per_scale": True, "cpu_only": True,
        "recognition_input_rule": "full canonical crops; detector-only resize",
    }, indent=2), encoding="utf-8")
    all_rows: list[dict[str, Any]] = []
    baselines: dict[str, dict[str, Any]] = {}
    ready_by_scale: dict[float, dict[str, Any]] = {}
    for scale in SCALES:
        limit = limits[str(scale)]
        scale_dir = output / f"scale_{scale:g}"
        scale_dir.mkdir()
        env = {**BASE_ENV, "TEXT_DETECTOR_LIMIT_SIDE_LEN": str(limit)}
        server = Server(argparse.Namespace(port=args.port, model_dir=args.model_dir, timeout=args.timeout), scale_dir, env)
        lifecycle = None
        try:
            ready = server.start()
            ready_by_scale[scale] = ready
            (scale_dir / "ready.json").write_text(json.dumps(ready, indent=2), encoding="utf-8")
            for kind in DOC_TYPES:
                selected = [document for document in documents if document.document_type == kind]
                warmup, _ = _post(kind, selected, args.port, args.timeout)
                (scale_dir / f"warmup_{kind}.json").write_text(json.dumps(warmup, indent=2), encoding="utf-8")
                for repeat in range(1, args.repeats + 1):
                    payload, client_seconds = _post(kind, selected, args.port, args.timeout)
                    row = _measurement(kind, selected, payload, client_seconds, repeat, scale, limit, baselines.get(kind))
                    all_rows.append(row)
                    raw_dir = scale_dir / "raw"
                    raw_dir.mkdir(exist_ok=True)
                    (raw_dir / f"{kind}_{repeat}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    if scale == 100.0 and repeat == 1:
                        baselines[kind] = row
                    server._sample()
        finally:
            lifecycle = server.stop()
            (scale_dir / "lifecycle.json").write_text(json.dumps(lifecycle, indent=2), encoding="utf-8")
        for row in all_rows:
            if row["scale_percent"] == scale:
                row["cleanup_verified"] = lifecycle.get("cleanup_verified") if lifecycle else False
                values = [value for value in (row.get("peak_rss_mb"), lifecycle.get("peak_process_memory_mb")) if value is not None]
                row["peak_rss_mb"] = max(values) if values else None
        print(f"scale {scale:g}% limit={limit}: cleanup={lifecycle.get('cleanup_verified') if lifecycle else False}", flush=True)
    (output / "raw_measurements.jsonl").write_text("\n".join(json.dumps(row) for row in all_rows) + "\n", encoding="utf-8")
    baseline_rows = [row for row in all_rows if row["scale_percent"] == 100.0]
    (output / "current_detector_resize_config.json").write_text(json.dumps({
        "effective_resize_configuration": ready_by_scale.get(100.0, {}).get("models", {}).get("text_detector", {}).get("resize"),
        "actual_baseline_tensor_shapes": sorted({json.dumps(shape) for row in baseline_rows for shape in row["detector_tensor_shapes"]}),
        "actual_baseline_tensor_pixel_counts": sorted({value for row in baseline_rows for value in row["detector_tensor_pixel_counts"]}),
    }, indent=2), encoding="utf-8")
    fields = sorted({key for row in all_rows for key, value in row.items() if not isinstance(value, (dict, list))})
    with (output / "raw_measurements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows({key: row.get(key) for key in fields} for row in all_rows)
    summary = []
    for scale in SCALES:
        for kind in DOC_TYPES:
            rows = [row for row in all_rows if row["scale_percent"] == scale and row["document_type"] == kind]
            summary.append({
                "scale_percent": scale, "detector_limit_side_len": limits[str(scale)], "document_type": kind,
                "actual_detector_tensor_shapes": json.dumps(rows[0]["detector_tensor_shapes"], separators=(",", ":")) if rows else None,
                "median_detector_tensor_pixel_count": _median(rows, "detector_tensor_pixel_count_total"),
                "median_detection_seconds": _median(rows, "detection_seconds"), "median_e2e_seconds": _median(rows, "e2e_seconds"),
                "median_docs_per_second": _median(rows, "docs_per_second"), "median_detected_line_count": _median(rows, "detected_line_count"),
                "median_recognition_candidate_crop_count": _median(rows, "recognition_candidate_crop_count"),
                "visible_exact": f"{int(statistics.median(row['visible_exact'] for row in rows))}/{rows[0]['visible_total']}" if rows else None,
                "visible_character_accuracy": _median(rows, "visible_character_accuracy"),
                "missing_fields": _median(rows, "missing_fields"), "extra_fields": _median(rows, "extra_fields"),
                "mrz_found": f"{rows[0]['mrz_found']}/{rows[0]['mrz_documents']}" if rows else None,
                "mrz_exact": f"{int(statistics.median(row['mrz_exact'] for row in rows))}/{rows[0]['mrz_documents']}" if rows else None,
                "failures": sum(row["failures"] for row in rows), "output_differences_vs_100": sum(row["output_difference_count_vs_100"] for row in rows),
                "peak_rss_mb": max((row["peak_rss_mb"] for row in rows if row.get("peak_rss_mb") is not None), default=None),
                "recognition_crop_content_identical_vs_100": all(row.get("recognition_crop_content_identical_vs_100") in (True, None) for row in rows),
                "cleanup_verified": all(row.get("cleanup_verified", False) for row in rows),
            })
    with (output / "scale_vs_speed_correctness.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summary[0])); writer.writeheader(); writer.writerows(summary)
    (output / "scale_vs_speed_correctness.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"completed: {output}", flush=True)
    return 0 if all(row.get("cleanup_verified") for row in all_rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
