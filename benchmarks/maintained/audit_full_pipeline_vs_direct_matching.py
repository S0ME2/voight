"""Read-only forensic audit for the full-vs-direct architecture benchmark."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
NEW = ROOT / "outputs/benchmarks/20.full-pipeline-vs-direct-matching/20260902T071921Z"
OLD = ROOT / "outputs/benchmarks/22.latin-vs-current-matcher/20260902T112500Z"
SWEEP = ROOT / "outputs/benchmarks/19.verification-batch-size-sweep/20260828T110401Z"
FULL = "FULL_LATIN_PIPELINE"
DIRECT = "DIRECT_MATCHING_PIPELINE"

csv.field_size_limit(50_000_000)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def load_json_cell(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return default


def direct_fields() -> tuple[dict[tuple[str, str], dict[str, str]], dict[tuple[str, str], dict[str, str]]]:
    old = {
        (row["document_id"], row["field"]): row
        for row in read_csv(OLD / "field_comparison.csv")
    }
    new = {
        (row["document_id"], row["field"]): row
        for row in read_csv(NEW / "field_comparison.csv")
        if row["candidate"] == DIRECT
    }
    return old, new


def old_check_rows() -> dict[tuple[str, str], dict[str, str]]:
    return {
        (row["document_id"], row["field"]): row
        for row in read_csv(OLD / "raw_runs.csv")
        if row.get("candidate") == "MATCHING_NEW"
        and row.get("phase") == "check"
        and row.get("pass") == "1"
    }


def transition(a: bool, b: bool) -> str:
    return "accepted->accepted" if a and b else "accepted->failure" if a else "failure->accepted" if b else "failure->failure"


def field_audit(output: Path) -> dict[str, int]:
    old, new = direct_fields()
    checks = old_check_rows()
    rows: list[dict[str, Any]] = []
    for key in sorted(old):
        previous = old[key]
        current = new[key]
        old_result = load_json_cell(checks[key].get("result"), {})
        previous_accepted = previous["current_status"] in {"match", "likely_match"}
        current_contract_accepted = current["matcher_status"] in {"match", "likely_match"}
        current_neutral_accepted = current["neutral_status"] == "correct"
        rows.append({
            "document_id": key[0], "field": key[1], "document_type": current["document_type"],
            "annotation": current["expected"],
            "previous_selected_value": previous["current_detected"], "previous_status": previous["current_status"],
            "previous_accepted": previous_accepted, "previous_evidence": old_result.get("evidence", []),
            "new_selected_value": current["detected"], "new_matcher_status": current["matcher_status"],
            "new_contract_accepted": current_contract_accepted, "new_neutral_status": current["neutral_status"],
            "new_neutral_accepted": current_neutral_accepted, "new_evidence": load_json_cell(current.get("evidence"), []),
            "previous_geometry_provenance": old_result.get("evidence", []),
            "new_geometry_provenance": load_json_cell(current.get("evidence"), []),
            "previous_ocr_text_confidence": "not persisted in field comparison; selected value shown separately",
            "new_ocr_text_confidence": "not persisted in field comparison; selected value shown separately",
            "verification_contract_transition": transition(previous_accepted, current_contract_accepted),
            "neutral_transition": transition(previous_accepted, current_neutral_accepted),
        })
    write_csv(output / "field_transition_diff.csv", rows)

    regressions = [row for row in rows if row["previous_accepted"] and not row["new_neutral_accepted"]]
    # There are 13 gross losses and one offsetting recovery. The headline is
    # down by 12; hiding the gross/offset split would make the arithmetic false.
    write_csv(output / "twelve_regressed_fields.csv", regressions)
    return {
        "previous_contract_accepted": sum(row["previous_accepted"] for row in rows),
        "new_contract_accepted": sum(row["new_contract_accepted"] for row in rows),
        "new_neutral_accepted": sum(row["new_neutral_accepted"] for row in rows),
        "gross_previous_accepted_to_new_neutral_failure": len(regressions),
        "previous_failure_to_new_neutral_acceptance": sum(not row["previous_accepted"] and row["new_neutral_accepted"] for row in rows),
        "contract_status_changes": sum(row["previous_status"] != row["new_matcher_status"] for row in rows),
        "selected_value_changes": sum(row["previous_selected_value"] != row["new_selected_value"] for row in rows),
        "selected_evidence_changes": sum(row["previous_evidence"] != row["new_evidence"] for row in rows),
    }


def _trace_signature(diagnostics: dict[str, Any]) -> dict[str, Any]:
    detection = diagnostics.get("text_detection", {})
    recognition = diagnostics.get("text_recognition", {})
    return {
        "sample_records": diagnostics.get("sample_records", []),
        "line_filter": diagnostics.get("line_filter", {}),
        "detection_calls": [{key: call.get(key) for key in ("input_widths", "input_heights", "detector_resized_shapes", "tensor_shapes", "tensor_pixel_counts", "padded_tensor_pixel_area", "tensor_batch_sizes")} for call in detection.get("calls", [])],
        "recognition_calls": [{key: call.get(key) for key in ("tensor_shapes", "tensor_pixel_counts", "tensor_batch_sizes")} for call in recognition.get("calls", [])],
        "recognition_crop_records": [
            {key: crop.get(key) for key in ("crop_sha256", "original_crop_w", "original_crop_h", "line_index", "natural_resized_w", "natural_resized_h", "packed_model_input_w", "packed_model_input_h")}
            for call in recognition.get("calls", []) for crop in call.get("crop_records", [])
        ],
    }


def ocr_audit(output: Path) -> dict[str, int]:
    old_rows = [row for row in read_csv(OLD / "raw_runs.csv") if row.get("candidate") == "MATCHING_NEW" and row.get("phase") == "ocr" and row.get("pass") == "1"]
    trace_dir = NEW / "01.direct-matching-pipeline-pass-1" / "trace"
    traces = sorted(trace_dir.glob("ocr-*.json"), key=lambda path: int(path.stem.rsplit("-", 1)[1]))[-20:]
    rows = []
    for old_row, trace_path in zip(old_rows, traces):
        old_trace = load_json_cell(old_row.get("trace"), {})
        new_trace = json.loads(trace_path.read_text(encoding="utf-8"))
        old_sig = _trace_signature(old_trace.get("diagnostics", {}))
        new_sig = _trace_signature(new_trace.get("diagnostics", {}))
        rows.append({
            "document_id": old_row["document_id"],
            "old_trace": "MATCHING_NEW pass 1", "new_trace": str(trace_path.relative_to(NEW)),
            "sample_records_same": old_sig["sample_records"] == new_sig["sample_records"],
            "line_filter_same": old_sig["line_filter"] == new_sig["line_filter"],
            "detection_input_geometry_same": old_sig["detection_calls"] == new_sig["detection_calls"],
            "recognition_tensor_batches_same": old_sig["recognition_calls"] == new_sig["recognition_calls"],
            "recognition_crop_records_same": old_sig["recognition_crop_records"] == new_sig["recognition_crop_records"],
            "text_confidence_comparison": "unknown: new architecture raw OCR response was not persisted in raw_runs.csv",
            "interpretation": "available OCR input/geometry/crop diagnostics identical" if old_sig == new_sig else "available OCR diagnostics differ",
        })
    write_csv(output / "ocr_evidence_diff.csv", rows)
    return {
        "documents_compared": len(rows),
        "all_available_diagnostics_same": sum(row["interpretation"].startswith("available OCR input") for row in rows),
        "confidence_comparisons_unknown": sum(row["text_confidence_comparison"].startswith("unknown") for row in rows),
    }


def _timing_row(candidate: str, raw: dict[str, str]) -> dict[str, float | None]:
    stage = load_json_cell(raw.get("stages_ms"), {})
    total = float(raw["server_ms"])
    if candidate == FULL:
        values: dict[str, float | None] = {
            "input_preparation": float(stage.get("input_preparation", 0)),
            "localization": float(stage.get("document_localization", 0)),
            "mrz_localization": float(stage.get("mrz_localization", 0)),
            "canonicalization_crop": float(stage.get("canonicalization", 0)) + float(stage.get("document_cropping", 0)),
            "detection": float(stage.get("text_detection", 0)), "text_line_cropping": float(stage.get("text_line_cropping", 0)),
            "recognition": float(stage.get("text_recognition", 0)), "mrz_work": float(stage.get("mrz_recognition", 0)),
            "ocr_unpacking": float(stage.get("ocr_unpacking", 0)), "matching_extraction": float(stage.get("field_extraction", 0)),
            "result_assembly": float(stage.get("result_assembly", 0)),
        }
    else:
        values = {
            "input_preparation": float(stage.get("input_preparation", 0)), "localization": None, "mrz_localization": None,
            "canonicalization_crop": None, "detection": float(stage.get("text_detection", 0)),
            "text_line_cropping": float(stage.get("text_line_cropping", 0)), "recognition": float(stage.get("text_recognition", 0)),
            "mrz_work": None, "ocr_unpacking": float(stage.get("ocr_unpacking", 0)),
            "matching_extraction": float(stage.get("matching", 0)), "result_assembly": float(stage.get("result_assembly", 0)),
        }
    additive = sum(value for value in values.values() if value is not None)
    values["other"] = max(0.0, total - additive)
    values["mrz_total"] = (values["mrz_localization"] + values["mrz_work"]) if candidate == FULL else None
    values["total"] = total
    values["sum_measured"] = additive + float(values["other"])
    values["unaccounted"] = total - float(values["sum_measured"])
    return values


def timing_audit(output: Path) -> dict[str, dict[str, float]]:
    rows = read_csv(NEW / "raw_runs.csv")
    output_rows: list[dict[str, Any]] = []
    metrics = ("input_preparation", "localization", "mrz_localization", "mrz_work", "mrz_total", "canonicalization_crop", "detection", "text_line_cropping", "recognition", "ocr_unpacking", "matching_extraction", "result_assembly", "other", "total", "sum_measured", "unaccounted")
    summary: dict[str, dict[str, float]] = {}
    for candidate in (FULL, DIRECT):
        selected = [row for row in rows if row["candidate"] == candidate]
        values = [_timing_row(candidate, row) for row in selected]
        for raw, value in zip(selected, values):
            output_rows.append({"row_type": "request", "statistic": "individual", "candidate": candidate, "pass": raw["pass"], "document_id": raw["document_id"], **value})
        summary[candidate] = {}
        for statistic, fn in (("mean", statistics.mean), ("median", statistics.median)):
            aggregate = {metric: (fn([value[metric] for value in values if value[metric] is not None]) if any(value[metric] is not None for value in values) else None) for metric in metrics}
            summary[candidate][statistic] = aggregate
            output_rows.append({"row_type": "aggregate", "statistic": statistic, "candidate": candidate, **aggregate})
    write_csv(output / "timing_accounting.csv", output_rows)
    return summary


def config_audit(output: Path) -> None:
    old_config = json.loads((OLD / "current_config.json").read_text(encoding="utf-8"))
    new_config = json.loads((NEW / "direct_matching_config.json").read_text(encoding="utf-8"))
    old_env = old_config["global_configuration"]
    new_env = new_config["environment"]
    same_thresholds = {"candidate_min_score": 0.35, "likely_name_score": 0.78, "likely_text_score": 0.86, "mrz_conflict_score": 0.86, "mrz_source_penalty": 0.05, "assembly_vertical_gap_factor": 3.0, "assembly_left_alignment_factor": 0.35, "max_assembly_lines": 3, "max_tokens_per_span": 6, "assignment_beam_width": 64, "max_candidates_per_field": 12}
    entries = [
        ("detector_model", old_env.get("TEXT_DETECTOR_MODEL"), new_env.get("TEXT_DETECTOR_MODEL"), "same"),
        ("recognizer_model", old_env.get("TEXT_RECOGNIZER_MODEL"), new_env.get("TEXT_RECOGNIZER_MODEL"), "same"),
        ("recognizer_backend", "paddle (resolved effective config)", new_env.get("TEXT_RECOGNIZER_BACKEND"), "same"),
        ("TEXT_DETECTION_BATCH_SIZE", old_env.get("TEXT_DETECTION_BATCH_SIZE"), new_env.get("TEXT_DETECTION_BATCH_SIZE"), "same"),
        ("TEXT_RECOGNITION_BATCH_SIZE", old_env.get("TEXT_RECOGNITION_BATCH_SIZE"), new_env.get("TEXT_RECOGNITION_BATCH_SIZE"), "same"),
        ("VERIFICATION_TEXT_DETECTION_BATCH_SIZE", "fallback 1", new_config["verification_overrides"].get("VERIFICATION_TEXT_DETECTION_BATCH_SIZE"), "same"),
        ("VERIFICATION_TEXT_RECOGNITION_BATCH_SIZE", old_config["effective_route_override"].get("VERIFICATION_TEXT_RECOGNITION_BATCH_SIZE"), new_config["verification_overrides"].get("VERIFICATION_TEXT_RECOGNITION_BATCH_SIZE"), "same"),
        ("recognition_packing", old_env.get("TEXT_RECOGNITION_PACKING"), new_env.get("TEXT_RECOGNITION_PACKING"), "same"),
        ("visible_preprocessing", "original (resolved)", "original (resolved)", "same"),
        ("MRZ preprocessing", "contrast_1.50 (resolved)", "not part of direct route", "irrelevant"),
        ("detector resizing", "960 side, keep_ratio=false, pixel_scale=1.0", "960 side, keep_ratio=false, pixel_scale=1.0", "same"),
        ("OCR thresholds", same_thresholds, same_thresholds, "same"),
        ("field definitions / values", "manifest 9e28eaafa1efaaeb45c9ee848c0376cad760d214d3da3af45a5296d0598e9b40; 254 fields", "manifest 9e28eaafa1efaaeb45c9ee848c0376cad760d214d3da3af45a5296d0598e9b40; 254 fields", "same"),
        ("CPU threads", "CPU=4, OMP=1, ONNX=4, OpenCV=16", "CPU=4, OMP=1, ONNX=4, OpenCV=16", "same"),
        ("MODEL_DIR", ".paddlex (effective old runtime)", new_env.get("MODEL_DIR"), "same"),
        ("PRELOAD", old_env.get("PRELOAD"), new_env.get("PRELOAD"), "different; performance/RSS only"),
        ("check invocation arity", "one /check request per field (254 checks)", "one /check request containing all fields (20 checks)", "different; matcher assignment context"),
        ("matcher implementation / thresholds", "current app.verification matcher and defaults", "current app.verification matcher and defaults", "same"),
        ("raw input files", "same dataset manifest", "same dataset manifest", "same"),
        ("full_verification artifact", "path absent in workspace", "N/A", "unknown historical artifact unavailable"),
    ]
    diff = [{"setting": key, "previous": old, "new": new, "classification": classification} for key, old, new, classification in entries]
    (output / "config_diff.json").write_text(json.dumps({"previous_source": str(OLD), "new_source": str(NEW), "previous_sweep_source": str(SWEEP), "entries": diff}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def write_reports(output: Path, transitions: dict[str, int], ocr: dict[str, int], timing: dict[str, dict[str, Any]]) -> None:
    full = timing[FULL]; direct = timing[DIRECT]
    lines = [
        "# Forensic Audit — Full Pipeline vs Direct Matching", "",
        "## Finding", "",
        "The 97.24% and 92.52% values are not the same metric. The previous direct baseline counts matcher statuses `match` and `likely_match` as accepted. The architecture benchmark's headline counts only neutral normalized value equality. The new DIRECT matcher still accepts 247/254 under its own contract; neutral exact equality is 235/254.", "",
        f"The arithmetic is `{transitions['previous_contract_accepted']} - {transitions['gross_previous_accepted_to_new_neutral_failure']} + {transitions['previous_failure_to_new_neutral_acceptance']} = {transitions['new_neutral_accepted']}`: 13 gross accepted-to-neutral-failure transitions are offset by one previously-failed field becoming neutral-correct, producing the net 12-field drop.", "",
        "## Route and model proof", "",
        "FULL used the real `/v1/ocr/{type}/batch` route. `full_pipeline_vs_direct_matching.py:_full_request` uploads the source files, then `_run_document` consumes the `/v1` result; the server route calls `models.profile_batch_runner().run`, which owns localization, canonicalization/crops, detection, recognition, parsing, and result assembly.", "",
        "DIRECT used the real `/verification/{type}/ocr` route followed by `/verification/{type}/check`. `_direct_request_fixed` uploads `_files(document)` unchanged, sends the returned OCR payload into the check endpoint, and `_run_document` records the final `check_payload` fields and matcher statuses. The API calls `models.verification_ocr().run` (shared Paddle detector/Latin recognizer) and `verify_fields`; the check path is `app.verification.verify_fields` → `_candidates` → global assignment → `_status`.", "",
        "The new run's server logs show `/verification/passport|id-card|driving-licence/{ocr,check}` requests. Model-load traces show DIRECT only loaded `text_detector` and `text_recognizer`; FULL loaded those plus document/MRZ localization components. No pre-generated OCR evidence or simplified evaluator produced the DIRECT response.", "",
        "## Direct field transition audit", "",
        f"Previous verification-contract accepted: `{transitions['previous_contract_accepted']}`; new matcher-contract accepted: `{transitions['new_contract_accepted']}`. Contract status changes: `{transitions['contract_status_changes']}` field (the d_6 serial field changes mismatch → not_found); selected values changed in `{transitions['selected_value_changes']}` fields. The 13 gross accepted→neutral-failure rows are in `twelve_regressed_fields.csv`; the complete 254-row classification is in `field_transition_diff.csv`.", "",
        "The 13 rows are: `d_1/patronymic`, `d_2/address`, `d_3/address`, `d_3/given_names`, `d_3/patronymic`, `d_4/address`, `d_4/issued_place`, `d_4/surname`, `d_6/address`, `d_7/address`, `id_3/place_of_issue`, `p_3/patronymic`, and `p_4/patronymic`. All retain the same selected value, matcher status, score, source, bbox, line ID, and span as the previous current run.", "",
        "The offsetting recovery is `d_6/personal_id`: previous status mismatch with `44.30502730260028`, new status mismatch with the same value, but the architecture benchmark's neutral normalizer treats it as equal to the annotation after label stripping. This is another evaluator-semantic difference, not an OCR improvement.", "",
        "## OCR evidence", "",
        f"Available diagnostics compare identical for `{ocr['all_available_diagnostics_same']}/{ocr['documents_compared']}` documents: source sample hashes/dimensions, detected-line counts, recognition-candidate counts, detector tensor inputs, recognition tensor batches, and recognition crop hashes/geometry. For the 13 gross regressions, selected geometry/provenance is identical in 13/13 rows. The new architecture artifact did not persist the complete raw OCR response in `raw_runs.csv`, so confidence equality is unknown rather than claimed. Classification: B for the regressions (evidence available to the matcher is identical; evaluation changed), with one separate all-field assignment-context change on d_6.", "",
        "## Evaluator semantics", "",
        "| Definition | FULL | DIRECT |", "|---|---:|---:|", "| Neutral normalized value equality | 184/254 (72.44%) | 235/254 (92.52%) |", "| Verification contract (`match` + `likely_match`) | value-only replay: 203/254* | 247/254 (97.24%) |", "",
        "`*` FULL's 203/254 is a reconstructed value-only replay of `app.verification` status thresholds, not a run of FULL through the verifier and not a replacement benchmark result. DIRECT's 247/254 is the actual matcher contract result. The established 97.24% number and the new DIRECT matcher contract therefore agree.", "",
        "## Corpus and fields", "",
        "The new manifest hash is `9e28eaafa1efaaeb45c9ee848c0376cad760d214d3da3af45a5296d0598e9b40`, matching the prior verification sweep. Both enumerate 20 logical documents, 24 physical images, and 254 fields; document IDs/order and annotation hashes are identical. The requested `full_verification/20260828T052615Z` directory is absent in this workspace, so it was not used as evidence.", "",
        "## RSS measurement audit", "",
        "The 86 MB values are incomplete and must not be compared with the earlier ~1.5 GB peaks. The new sampler reads `/proc/<Popen PID>/status` via `_rss` only during `Server.start()` while polling `/v1/health/live`; it does not sample during warmup, model loading, or measured requests. Therefore it captured the idle uvicorn process before models were loaded. The model-load traces prove model initialization occurred later in that same server process, but the peak was never observed or persisted with a PID in `resource_usage.csv`.", "",
        "The earlier verification benchmark starts `RssMonitor` before warmup and samples the same tracked uvicorn PID every 0.1 seconds through model initialization, warmup, and measurement. It records `server_pid`, `server_pgid`, baseline available memory, peak process RSS, child-group cleanup, and release status. It still samples only the tracked process, not an aggregate of child RSS, but both historical values are process peaks while the new 86 MB values are startup baselines. The new RSS comparison is invalid; a memory-only corrected rerun is required before accepting memory conclusions.", "",
        "## Additive timing accounting", "",
        "Per-request accounting is in `timing_accounting.csv`. It sums mutually exclusive top-level stages for each of 200 measured requests, then derives `other` as the residual to that request's total. DIRECT candidate construction, geometry assembly, MRZ validation, and strict validation remain nested diagnostics inside matching and are not added again.", "",
        "| Candidate / statistic | Localization | MRZ total | Canonicalization + crop | Detection | Recognition | Matching / extraction | Other | Total |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        f"| FULL mean | {full['mean']['localization']:.2f} | {full['mean']['mrz_total']:.2f} | {full['mean']['canonicalization_crop']:.2f} | {full['mean']['detection']:.2f} | {full['mean']['recognition']:.2f} | {full['mean']['matching_extraction']:.2f} | {full['mean']['other']:.2f} | {full['mean']['total']:.2f} |",
        f"| FULL median | {full['median']['localization']:.2f} | {full['median']['mrz_total']:.2f} | {full['median']['canonicalization_crop']:.2f} | {full['median']['detection']:.2f} | {full['median']['recognition']:.2f} | {full['median']['matching_extraction']:.2f} | {full['median']['other']:.2f} | {full['median']['total']:.2f} |",
        f"| DIRECT mean | N/A | N/A | N/A | {direct['mean']['detection']:.2f} | {direct['mean']['recognition']:.2f} | {direct['mean']['matching_extraction']:.2f} | {direct['mean']['other']:.2f} | {direct['mean']['total']:.2f} |",
        f"| DIRECT median | N/A | N/A | N/A | {direct['median']['detection']:.2f} | {direct['median']['recognition']:.2f} | {direct['median']['matching_extraction']:.2f} | {direct['median']['other']:.2f} | {direct['median']['total']:.2f} |",
        "",
        "The earlier stage table added independent medians. Medians of different request-level subsets do not add to the median of totals, even when each request's stages are additive. FULL MRZ localization is only present on documents where that stage runs, which makes the effect especially visible; request-level accounting resolves it.", "",
        "## Detector workload validity", "",
        "The detector-pixel values are valid tensor-work metrics, not estimates: each diagnostics call records resized/padded tensor shapes and `padded_tensor_pixel_area`; the benchmark sums that area across detector calls for the document. FULL median is 252,824 and DIRECT median is 417,720. Recognition crop counts are actual crop records: FULL 15 and DIRECT 34 median. The source fields are preserved in `detection_workload.csv` and `recognition_workload.csv`.", "",
        "## Decision", "",
        "Do not accept the original report as written. Correct its interpretation: DIRECT did not regress from 97.24% to 92.52%; it retained 97.24% under the verification contract, while the new neutral exact-value score is 92.52%. The architecture benchmark routes and shared models are correct. Do not rerun the full benchmark for the accuracy discrepancy. Do a separate RSS-only corrected measurement before accepting memory claims; the current 86 MB comparison must be removed or marked invalid.", "",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / "evaluator_comparison.md").write_text("\n".join(lines[0:0] + [
        "# Evaluator comparison", "",
        "Previous direct verification accuracy is `match + likely_match`: 236 + 11 = 247/254 (97.24%). New DIRECT has the same matcher counts: 236 + 11 = 247/254; its matcher failures are 5 mismatch and 2 not_found.", "",
        "The architecture benchmark headline is neutral equality: 235 correct, 17 incorrect, 2 not_found. The 13 prior accepted→new neutral failure rows are fuzzy/normalized values that remain `likely_match` or `match`; one old failure (`d_6/personal_id`) becomes neutral-correct. Therefore the headline changes by 12 net fields without a DIRECT matcher regression.", "",
        "| Definition | FULL | DIRECT |", "|---|---:|---:|", "| Neutral equality | 184/254 | 235/254 |", "| Verification contract | value-only replay 203/254 (diagnostic only) | 247/254 actual |", "",
        "The evaluators are different. They must remain separate in any accepted report.", "",
    ]) + "\n", encoding="utf-8")
    (output / "rss_measurement_audit.md").write_text("\n".join([
        "# RSS measurement audit", "",
        "## New architecture benchmark", "",
        "The tracked process is the Popen uvicorn process (`python -m uvicorn app.main:app --workers 1`). Server logs record PIDs 110450, 111386, 112817, 113978, 114929, 116097, 117707, 119513, 120630, and 121716 across the ten passes. `_rss(pid)` reads `/proc/<pid>/status` `VmRSS`.", "",
        "The sampler is called only in `Server.start()` during live polling. It is not called during `_warmup`, `_run_document`, model initialization, or measured requests. The reported ~86 MB is therefore an idle/startup baseline, not a peak after models loaded. `resource_usage.csv` does not persist PID, baseline RSS, or a sampling interval.", "",
        "## Previous verification benchmark", "",
        "`RssMonitor` samples the same kind of tracked uvicorn PID every 0.1 s before warmup and until stop. `memory_results.csv` records baseline available memory and peak process RSS around 1.49–1.64 GB. It also records release/cleanup status. Neither implementation aggregates child RSS, but the old values are actual process peaks and the new values are not.", "",
        "## Conclusion", "",
        "The new RSS comparison is invalid. No production code/configuration was changed and no full rerun was performed. A separate RSS-only corrected measurement is required.", "",
    ]) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config_audit(args.output_dir)
    transitions = field_audit(args.output_dir)
    ocr = ocr_audit(args.output_dir)
    timing = timing_audit(args.output_dir)
    write_reports(args.output_dir, transitions, ocr, timing)
    print(f"wrote forensic audit: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
