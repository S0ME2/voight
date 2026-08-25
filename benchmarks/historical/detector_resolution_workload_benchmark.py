"""Benchmark real Paddle detector tensor workloads with full-resolution OCR crops."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.historical.detector_resolution_benchmark import _measurement
from benchmarks.maintained.model_matrix_benchmark import Server, _post
from benchmarks.maintained.pipeline_breakdown import DOC_TYPES, validate_and_manifest

BROAD = (100.0, 85.0, 70.0, 55.0, 40.0)
BASE_ENV = {
    "RUNTIME_TARGET": "cpu", "OCR_DEVICE": "cpu", "CPU_THREADS": "4",
    "LOCALIZATION_BATCH_SIZE": "16", "TEXT_DETECTION_BATCH_SIZE": "16",
    "TEXT_RECOGNITION_BATCH_SIZE": "32", "MRZ_RECOGNITION_BATCH_SIZE": "16",
    "TEXT_RECOGNITION_PROCESSES": "1", "TEXT_RECOGNITION_PACKING": "aspect-ratio",
    "TEXT_DETECTOR_MODEL": "PP-OCRv6_medium_det", "TEXT_RECOGNIZER_MODEL": "latin_PP-OCRv5_mobile_rec",
    "DOCALIGNER_MODEL": "fastvit_sa24", "DOCALIGNER_MODEL_TYPE": "heatmap",
    "MRZ_RECOGNIZER_BACKEND": "generic-paddle", "MRZ_RECOGNIZER_MODEL": "20250221",
    "TEXT_DETECTOR_LIMIT_SIDE_LEN": "",
}


def _actual(payload: dict[str, Any]) -> dict[str, Any]:
    stage = payload.get("diagnostics", {}).get("text_detection", {})
    calls = stage.get("calls", [])
    shapes = [shape for call in calls for shape in call.get("tensor_shapes", [])]
    pixels = [value for call in calls for value in call.get("tensor_pixel_counts", [])]
    return {"shapes": shapes, "shape_signature": json.dumps(shapes, separators=(",", ":")), "pixels": pixels, "pixel_total": sum(pixels)}


def _source_signature(payload: dict[str, Any]) -> list[tuple[Any, ...]]:
    return sorted(
        (row.get("sample_id"), row.get("width"), row.get("height"), row.get("crop_sha256"))
        for row in payload.get("diagnostics", {}).get("text_recognition", {}).get("sample_records", [])
    )


def _scale_for_target(target_pct: float) -> float:
    return math.sqrt(target_pct / 100.0)


def _fmt(value: float) -> str:
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _clean(row: dict[str, Any], baseline: dict[str, Any], *, include_mrz: bool) -> bool:
    same_visible = (
        row["visible_exact"] == baseline["visible_exact"]
        and row["missing_fields"] <= baseline["missing_fields"]
        and row["extra_fields"] <= baseline["extra_fields"]
        and row["visible_character_accuracy"] >= baseline["visible_character_accuracy"] - 1e-12
    )
    mrz_ok = not include_mrz or (
        row["mrz_found"] >= baseline["mrz_found"] and row["mrz_exact"] >= baseline["mrz_exact"]
    )
    return same_visible and mrz_ok and row["failures"] == 0


def _summarize(rows: list[dict[str, Any]], baselines: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for requested in sorted({row["requested_pixel_target_pct"] for row in rows}, reverse=True):
        for kind in DOC_TYPES:
            selected = [row for row in rows if row["requested_pixel_target_pct"] == requested and row["document_type"] == kind and row["status"] == "ok"]
            if not selected:
                continue
            baseline_pixels = baselines[kind]["detector_tensor_pixel_count_total"]
            actual_pixels = statistics.median(row["detector_tensor_pixel_count_total"] for row in selected)
            result.append({
                "requested_pixel_target_pct": requested,
                "document_type": kind,
                "actual_tensor_shapes": json.dumps(selected[0]["detector_tensor_shapes"], separators=(",", ":")),
                "actual_detector_pixels": int(actual_pixels),
                "actual_pixel_ratio_vs_baseline_pct": 100 * actual_pixels / baseline_pixels,
                "median_detection_seconds": statistics.median(row["detection_seconds"] for row in selected),
                "median_e2e_seconds": statistics.median(row["e2e_seconds"] for row in selected),
                "median_docs_per_second": statistics.median(row["docs_per_second"] for row in selected),
                "median_detected_line_count": statistics.median(row["detected_line_count"] for row in selected),
                "median_recognition_crop_count": statistics.median(row["recognition_candidate_crop_count"] for row in selected),
                "visible_exact": f"{int(statistics.median(row['visible_exact'] for row in selected))}/{selected[0]['visible_total']}",
                "visible_character_accuracy_pct": 100 * statistics.median(row["visible_character_accuracy"] for row in selected),
                "missing_fields": statistics.median(row["missing_fields"] for row in selected),
                "extra_fields": statistics.median(row["extra_fields"] for row in selected),
                "mrz_found": f"{int(statistics.median(row['mrz_found'] for row in selected))}/{selected[0]['mrz_documents']}",
                "mrz_exact": f"{int(statistics.median(row['mrz_exact'] for row in selected))}/{selected[0]['mrz_documents']}",
                "output_differences_vs_100": sum(row["output_difference_count_vs_100"] for row in selected),
                "failures": sum(row["failures"] for row in selected),
                "peak_rss_mb": max(row["peak_rss_mb"] for row in selected if row.get("peak_rss_mb") is not None),
                "canonical_source_identical_vs_100": all(row["canonical_source_identical_vs_100"] for row in selected),
                "recognition_crop_geometry_changed_vs_100": any(row["recognition_crop_content_identical_vs_100"] is False for row in selected),
                "crop_geometry_unexpected": any(row["crop_geometry_unexpected"] for row in selected),
                "cleanup_verified": all(row.get("cleanup_verified", False) for row in selected),
            })
    return sorted(result, key=lambda row: (row["document_type"], row["actual_detector_pixels"]))


def _refinement_targets(broad_rows: list[dict[str, Any]]) -> tuple[list[float], dict[str, Any]]:
    primary = ("id_card", "driving_license")
    boundaries = {}
    targets = []
    for kind in primary:
        values = sorted(
            {row["requested_pixel_target_pct"] for row in broad_rows if row["document_type"] == kind},
            reverse=True,
        )
        baseline = next(row for row in broad_rows if row["document_type"] == kind and row["requested_pixel_target_pct"] == 100.0)
        clean_target = 100.0
        bad_target = None
        for target in values:
            row = next(row for row in broad_rows if row["document_type"] == kind and row["requested_pixel_target_pct"] == target)
            if _clean(row, baseline, include_mrz=kind == "id_card"):
                clean_target = target
            else:
                bad_target = target
                break
        if bad_target is None:
            bad_target = max(20.0, min(values) - 10.0)
            clean_target = min(values)
            boundaries[kind] = {"clean_broad_pct": clean_target, "degraded_broad_pct": None, "note": "no visible-gate degradation in broad sweep"}
        else:
            boundaries[kind] = {"clean_broad_pct": clean_target, "degraded_broad_pct": bad_target, "note": "first lower broad target failing the visible gate"}
        gap = clean_target - bad_target
        targets.extend((clean_target - gap * 0.25, clean_target - gap * 0.50, clean_target - gap * 0.75))
    unique = sorted({round(target, 3) for target in targets if target > 0 and target not in BROAD}, reverse=True)
    return unique, boundaries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/benchmark")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/benchmarks/detector_resolution_workload")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 3:
        raise SystemExit("--repeats must be at least 3")
    documents, manifest = validate_and_manifest(args.dataset_root)
    output = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps({
        "dataset": manifest, "models": BASE_ENV, "broad_requested_pixel_targets_pct": BROAD,
        "target_definition": "sqrt(pixel fraction) applied to each actual post-DetResizeForTest H/W, then each dimension rounded to nearest multiple of 32",
        "thresholds_fixed": {"text_det_thresh": 0.30, "text_det_box_thresh": 0.50, "text_det_unclip_ratio": 2.0},
        "warmup": 1, "measured_repeats": args.repeats, "fresh_server_per_real_target": True,
        "recognition_rule": "full-resolution canonical source image; detector-only tensor scaling",
    }, indent=2), encoding="utf-8")
    all_rows: list[dict[str, Any]] = []
    baselines: dict[str, dict[str, Any]] = {}
    seen_shapes: dict[str, list[str]] = {}
    ready_by_target: dict[float, dict[str, Any]] = {}
    skipped: list[dict[str, Any]] = []

    def run_target(target_pct: float, phase: str) -> None:
        dimension_scale = _scale_for_target(target_pct)
        label = f"{phase}_{_fmt(target_pct)}"
        target_dir = output / label
        target_dir.mkdir()
        env = {**BASE_ENV, "TEXT_DETECTOR_PIXEL_SCALE": _fmt(dimension_scale)}
        server = Server(argparse.Namespace(port=args.port, model_dir=args.model_dir, timeout=args.timeout), target_dir, env)
        lifecycle: dict[str, Any] = {}
        target_rows: list[dict[str, Any]] = []
        try:
            ready = server.start()
            ready_by_target[target_pct] = ready
            (target_dir / "ready.json").write_text(json.dumps(ready, indent=2), encoding="utf-8")
            for kind in DOC_TYPES:
                selected = [document for document in documents if document.document_type == kind]
                warmup, _ = _post(kind, selected, args.port, args.timeout)
                warm_actual = _actual(warmup)
                (target_dir / f"warmup_{kind}.json").write_text(json.dumps(warmup, indent=2), encoding="utf-8")
                prior_values = seen_shapes.get(kind, [])
                materially_different = warm_actual["shape_signature"] not in prior_values
                verification = {
                    "document_type": kind, "requested_pixel_target_pct": target_pct,
                    "dimension_scale": dimension_scale, "actual": warm_actual,
                    "previous_actual_tensor_shapes": prior_values,
                    "materially_different_from_previous": materially_different,
                }
                (target_dir / f"verification_{kind}.json").write_text(json.dumps(verification, indent=2), encoding="utf-8")
                if not materially_different:
                    skipped.append({**verification, "phase": phase, "status": "skipped_duplicate_tensor"})
                    continue
                for repeat in range(1, args.repeats + 1):
                    payload, client_seconds = _post(kind, selected, args.port, args.timeout)
                    row = _measurement(kind, selected, payload, client_seconds, repeat, target_pct, dimension_scale, baselines.get(kind))
                    row.update({
                        "phase": phase, "requested_pixel_target_pct": target_pct,
                        "dimension_scale": dimension_scale,
                        "actual_pixel_ratio_vs_baseline_pct": None if kind not in baselines else 100 * row["detector_tensor_pixel_count_total"] / baselines[kind]["detector_tensor_pixel_count_total"],
                        "canonical_source_signature": _source_signature(payload),
                    })
                    row["canonical_source_identical_vs_100"] = baselines.get(kind, {}).get("canonical_source_signature") in (None, row["canonical_source_signature"])
                    row["recognition_crop_geometry_changed_vs_100"] = row.get("recognition_crop_content_identical_vs_100") is False
                    row["crop_geometry_unexpected"] = not row["canonical_source_identical_vs_100"]
                    target_rows.append(row)
                    raw_dir = target_dir / "raw"
                    raw_dir.mkdir(exist_ok=True)
                    (raw_dir / f"{kind}_{repeat}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    if target_pct == 100.0 and repeat == 1:
                        baselines[kind] = row
                    server._sample()
                seen_shapes.setdefault(kind, []).append(warm_actual["shape_signature"])
        finally:
            lifecycle = server.stop()
            (target_dir / "lifecycle.json").write_text(json.dumps(lifecycle, indent=2), encoding="utf-8")
        for row in target_rows:
            row["cleanup_verified"] = lifecycle.get("cleanup_verified", False)
            values = [value for value in (row.get("peak_rss_mb"), lifecycle.get("peak_process_memory_mb")) if value is not None]
            row["peak_rss_mb"] = max(values) if values else None
        all_rows.extend(target_rows)
        print(f"{phase} {target_pct:g}% side={dimension_scale:.6f}: cleanup={lifecycle.get('cleanup_verified')}", flush=True)

    for target in BROAD:
        run_target(target, "broad")

    broad_summary = _summarize(all_rows, baselines)
    broad_by_kind = {kind: next(row for row in broad_summary if row["document_type"] == kind and row["requested_pixel_target_pct"] == 100.0) for kind in DOC_TYPES}
    broad_for_boundary = []
    for row in broad_summary:
        expanded = dict(row)
        expanded.update({
            "visible_exact": int(row["visible_exact"].split("/", 1)[0]),
            "visible_total": int(row["visible_exact"].split("/", 1)[1]),
            "visible_character_accuracy": row["visible_character_accuracy_pct"] / 100,
            "mrz_found": int(row["mrz_found"].split("/", 1)[0]),
            "mrz_documents": int(row["mrz_found"].split("/", 1)[1]),
            "mrz_exact": int(row["mrz_exact"].split("/", 1)[0]),
            "missing_fields": row["missing_fields"], "extra_fields": row["extra_fields"],
            "failures": row["failures"],
        })
        broad_for_boundary.append(expanded)
    fine_targets, boundary_info = _refinement_targets(broad_for_boundary)
    (output / "boundary_selection.json").write_text(json.dumps({"boundaries": boundary_info, "fine_requested_pixel_targets_pct": fine_targets}, indent=2), encoding="utf-8")
    for target in fine_targets:
        run_target(target, "refine")

    summaries = _summarize(all_rows, baselines)
    broad_rows = [row for row in summaries if row["requested_pixel_target_pct"] in BROAD]
    refine_rows = [row for row in summaries if row["requested_pixel_target_pct"] not in BROAD]
    for name, rows in (("broad_sweep.csv", broad_rows), ("boundary_refinement.csv", refine_rows)):
        with (output / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["document_type"])
            writer.writeheader(); writer.writerows(rows)
    (output / "broad_sweep.json").write_text(json.dumps(broad_rows, indent=2), encoding="utf-8")
    (output / "boundary_refinement.json").write_text(json.dumps(refine_rows, indent=2), encoding="utf-8")
    (output / "skipped_candidates.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    (output / "baseline_tensor_dimensions.json").write_text(json.dumps({
        kind: {"actual_tensor_shapes": baselines[kind]["detector_tensor_shapes"], "actual_tensor_pixel_counts": baselines[kind]["detector_tensor_pixel_counts"], "actual_detector_pixels_total": baselines[kind]["detector_tensor_pixel_count_total"]}
        for kind in DOC_TYPES
    }, indent=2), encoding="utf-8")
    (output / "tested_tensor_dimensions.json").write_text(json.dumps({
        kind: [{"requested_pixel_target_pct": row["requested_pixel_target_pct"], "actual_tensor_shapes": json.loads(row["actual_tensor_shapes"]), "actual_detector_pixels_total": row["actual_detector_pixels"]} for row in summaries if row["document_type"] == kind]
        for kind in DOC_TYPES
    }, indent=2), encoding="utf-8")
    (output / "raw_measurements.jsonl").write_text("\n".join(json.dumps(row) for row in all_rows) + "\n", encoding="utf-8")
    scalar_fields = sorted({key for row in all_rows for key, value in row.items() if not isinstance(value, (dict, list))})
    with (output / "raw_measurements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=scalar_fields); writer.writeheader(); writer.writerows({key: row.get(key) for key in scalar_fields} for row in all_rows)
    (output / "effective_baseline_resize_config.json").write_text(json.dumps(ready_by_target.get(100.0, {}).get("models", {}).get("text_detector", {}).get("resize"), indent=2), encoding="utf-8")
    print(f"completed: {output}", flush=True)
    return 0 if all(row.get("cleanup_verified") for row in all_rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
