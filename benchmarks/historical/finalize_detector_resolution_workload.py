"""Finalize a completed detector workload run after an interrupted report step."""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.historical.detector_resolution_benchmark import _measurement
from benchmarks.historical.detector_resolution_workload_benchmark import _actual, _source_signature, _summarize
from benchmarks.maintained.pipeline_breakdown import DOC_TYPES, validate_and_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    args = parser.parse_args()
    run = args.run.resolve()
    documents, _ = validate_and_manifest(args.dataset_root)
    baselines = {}
    rows = []
    targets = sorted(run.glob("broad_*/raw")) + sorted(run.glob("refine_*/raw"))
    for raw_dir in targets:
        match = re.match(r"(?:broad|refine)_([0-9.]+)$", raw_dir.parent.name)
        if not match:
            continue
        phase = raw_dir.parent.name.split("_", 1)[0]
        requested = float(match.group(1))
        lifecycle = json.loads((raw_dir.parent / "lifecycle.json").read_text())
        for raw in sorted(raw_dir.glob("*.json")):
            kind = raw.stem.rsplit("_", 1)[0]
            if kind not in DOC_TYPES:
                continue
            selected = [document for document in documents if document.document_type == kind]
            payload = json.loads(raw.read_text())
            row = _measurement(kind, selected, payload, payload.get("total_seconds", 0.0), int(raw.stem.rsplit("_", 1)[1]), requested, (requested / 100) ** 0.5, baselines.get(kind))
            row.update({"phase": phase, "requested_pixel_target_pct": requested, "dimension_scale": (requested / 100) ** 0.5, "canonical_source_signature": _source_signature(payload)})
            row["actual_pixel_ratio_vs_baseline_pct"] = None if kind not in baselines else 100 * row["detector_tensor_pixel_count_total"] / baselines[kind]["detector_tensor_pixel_count_total"]
            row["canonical_source_identical_vs_100"] = baselines.get(kind, {}).get("canonical_source_signature") in (None, row["canonical_source_signature"])
            row["recognition_crop_geometry_changed_vs_100"] = row.get("recognition_crop_content_identical_vs_100") is False
            row["crop_geometry_unexpected"] = not row["canonical_source_identical_vs_100"]
            row["cleanup_verified"] = lifecycle.get("cleanup_verified", False)
            values = [value for value in (row.get("peak_rss_mb"), lifecycle.get("peak_process_memory_mb")) if value is not None]
            row["peak_rss_mb"] = max(values) if values else None
            rows.append(row)
            if requested == 100.0 and kind not in baselines:
                baselines[kind] = row
    rows.sort(key=lambda row: (0 if row["phase"] == "broad" else 1, -row["requested_pixel_target_pct"], DOC_TYPES.index(row["document_type"]), row["repeat"]))
    accepted = []
    duplicate_rows = []
    seen: dict[str, list[str]] = {kind: [] for kind in DOC_TYPES}
    candidate_keys = sorted({(row["phase"], row["requested_pixel_target_pct"], row["document_type"]) for row in rows}, key=lambda key: (0 if key[0] == "broad" else 1, -key[1], DOC_TYPES.index(key[2])))
    for phase, requested, kind in candidate_keys:
        group = [row for row in rows if row["phase"] == phase and row["requested_pixel_target_pct"] == requested and row["document_type"] == kind]
        pixels = group[0]["detector_tensor_pixel_count_total"]
        shape_signature = json.dumps(group[0]["detector_tensor_shapes"], separators=(",", ":"))
        duplicate = shape_signature in seen[kind]
        for row in group:
            row["accepted_tensor_target"] = not duplicate
            row["duplicate_tensor_target"] = duplicate
        if duplicate:
            duplicate_rows.append({"phase": phase, "requested_pixel_target_pct": requested, "document_type": kind, "actual_detector_pixels": pixels, "status": "excluded_duplicate_tensor"})
        else:
            accepted.extend(group)
            seen[kind].append(shape_signature)
    summaries = _summarize(accepted, baselines)
    broad = [row for row in summaries if row["requested_pixel_target_pct"] in (100.0, 85.0, 70.0, 55.0, 40.0)]
    refine = [row for row in summaries if row["requested_pixel_target_pct"] not in (100.0, 85.0, 70.0, 55.0, 40.0)]
    for name, values in (("broad_sweep.csv", broad), ("boundary_refinement.csv", refine)):
        with (run / name).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]) if values else ["document_type"]); writer.writeheader(); writer.writerows(values)
    (run / "broad_sweep.json").write_text(json.dumps(broad, indent=2), encoding="utf-8")
    (run / "boundary_refinement.json").write_text(json.dumps(refine, indent=2), encoding="utf-8")
    (run / "raw_measurements.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    fields = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))})
    with (run / "raw_measurements.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows({key: row.get(key) for key in fields} for row in rows)
    verification_skips = []
    for verification in sorted(run.glob("*/verification_*.json")):
        data = json.loads(verification.read_text())
        if data.get("materially_different_from_previous") is False:
            verification_skips.append({**data, "status": "skipped_duplicate_tensor_at_run_time"})
    (run / "skipped_candidates.json").write_text(json.dumps(verification_skips + duplicate_rows, indent=2), encoding="utf-8")
    (run / "baseline_tensor_dimensions.json").write_text(json.dumps({kind: {"actual_tensor_shapes": baselines[kind]["detector_tensor_shapes"], "actual_tensor_pixel_counts": baselines[kind]["detector_tensor_pixel_counts"], "actual_detector_pixels_total": baselines[kind]["detector_tensor_pixel_count_total"]} for kind in DOC_TYPES}, indent=2), encoding="utf-8")
    (run / "tested_tensor_dimensions.json").write_text(json.dumps({kind: [{"requested_pixel_target_pct": row["requested_pixel_target_pct"], "actual_tensor_shapes": json.loads(row["actual_tensor_shapes"]), "actual_detector_pixels_total": row["actual_detector_pixels"]} for row in summaries if row["document_type"] == kind] for kind in DOC_TYPES}, indent=2), encoding="utf-8")
    ready = json.loads((run / "05.broad-100" / "ready.json").read_text())
    (run / "effective_baseline_resize_config.json").write_text(json.dumps(ready.get("models", {}).get("text_detector", {}).get("resize"), indent=2), encoding="utf-8")
    print(run)
    print(f"rows={len(rows)} summaries={len(summaries)} cleanup={all(row['cleanup_verified'] for row in rows)}")
    return 0 if rows and all(row["cleanup_verified"] for row in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
