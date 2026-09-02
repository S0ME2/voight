"""Reconcile fixed-crop and real-pipeline MRZ preprocessing benchmarks."""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import Settings
from app.imaging import preprocess_variant
from app.inference.packing import recognition_batch_packer
from app.models import Models
from benchmarks.maintained.model_matrix_benchmark import Server, _post, _score, _stage_totals
from benchmarks.maintained.pipeline_breakdown import DOC_TYPES, annotation_truth, validate_and_manifest

VARIANTS = ("original", "contrast_1.30", "contrast_1.50", "sharpen_light")
RUN_ORDER = ("baseline_1", "sharpen_light", "baseline_2", "contrast_1.30", "baseline_3", "contrast_1.50")
BASE_ENV = {
    "RUNTIME_TARGET": "cpu", "OCR_DEVICE": "cpu", "PRELOAD": "true", "LOGGING": "false",
    "CPU_THREADS": "4", "LOCALIZATION_BATCH_SIZE": "4", "TEXT_DETECTION_BATCH_SIZE": "1",
    "TEXT_RECOGNITION_BATCH_SIZE": "2", "MRZ_RECOGNITION_BATCH_SIZE": "2",
    "TEXT_RECOGNITION_PROCESSES": "1", "TEXT_RECOGNITION_PACKING": "fixed-width",
    "TEXT_DETECTOR_MODEL": "PP-OCRv6_medium_det", "TEXT_RECOGNIZER_MODEL": "latin_PP-OCRv5_mobile_rec",
    "DOCALIGNER_MODEL": "fastvit_sa24", "DOCALIGNER_MODEL_TYPE": "heatmap",
    "MRZ_RECOGNIZER_BACKEND": "generic-paddle", "MRZ_RECOGNIZER_MODEL": "20250221",
    "OCR_MAX_SIDE": "3000", "OCR_CONTRAST": "1.25", "MRZ_POLYGON_PADDING_RATIO": "0.03",
    "DOCALIGNER_PADDING": "100", "TEXT_DETECTOR_LIMIT_SIDE_LEN": "960",
}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(row for row in rows)


def _source_hash(path: Path) -> str:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode {path}")
    import hashlib
    return hashlib.sha256(image.tobytes()).hexdigest()


def _truth(document: Any) -> dict[str, Any]:
    return annotation_truth(document)


def _distance(left: str, right: str) -> int:
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        next_row = [i]
        for j, b in enumerate(right, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = next_row
    return row[-1]


def _doc_mrz(document: Any, item: dict[str, Any]) -> dict[str, Any]:
    expected = [line for line in _truth(document).get("mrz", {}).get("lines", []) if isinstance(line, str)]
    result = item.get("result") or {}
    mrz = result.get("mrz") or {}
    actual = list(mrz.get("raw_lines") or [])
    matches = [index < len(actual) and actual[index] == value for index, value in enumerate(expected)]
    errors = sum(_distance(expected_line, actual[index] if index < len(actual) else "") for index, expected_line in enumerate(expected))
    validations = list(mrz.get("validations") or [])
    return {
        "document_id": document.document_id,
        "document_type": document.document_type,
        "recognized_raw_mrz_text": "\n".join(actual),
        "recognized_mrz_lines": actual,
        "expected_mrz": expected,
        "exact_line_matches": matches,
        "exact_lines": sum(matches),
        "line_total": len(expected),
        "mrz_character_errors": errors,
        "parsed_fields": mrz.get("fields") or {},
        "icao_check_digit_validations": validations,
        "whole_mrz_exact": actual == expected,
    }


def _request_rows(kind: str, documents: list[Any], payload: dict[str, Any], client_seconds: float, run: str, repeat: int) -> list[dict[str, Any]]:
    rows = []
    for document, item in zip(documents, payload.get("items", [])):
        detail = _doc_mrz(document, item) if document.document_type in {"passport", "id_card"} else None
        rows.append({
            "run": run, "repeat": repeat, "document_id": document.document_id,
            "document_type": document.document_type, "client_e2e_seconds": client_seconds,
            "server_e2e_seconds": payload.get("total_seconds"),
            "mrz": detail,
            "success": item.get("success", False),
        })
    return rows


def _run_preflight(args: argparse.Namespace, documents: list[Any], output: Path) -> dict[str, Any]:
    directory = output / "preflight"
    trace_dir = directory / "trace"
    directory.mkdir(parents=True, exist_ok=True)
    env = {**BASE_ENV, "MODEL_DIR": str(args.model_dir), "VOIGHT_BENCHMARK_MRZ_TRACE": str(trace_dir),
           "VOIGHT_BENCHMARK_MRZ_PREPROCESSING": "original", "VOIGHT_BENCHMARK_MRZ_CROP_PREPROCESSING": "",
           "VOIGHT_BENCHMARK_MRZ_PREPROCESS_BEFORE_PACKING": "1"}
    server = Server(argparse.Namespace(port=args.port, model_dir=args.model_dir, timeout=args.timeout), directory, env)
    payloads = {}
    lifecycle = None
    try:
        server.start()
        for kind in DOC_TYPES:
            payload, elapsed = _post(kind, [doc for doc in documents if doc.document_type == kind], args.port, args.timeout)
            payloads[kind] = {"payload": payload, "client_seconds": elapsed}
            write_json(directory / f"{kind}.json", payload)
    finally:
        lifecycle = server.stop()
        write_json(directory / "lifecycle.json", lifecycle)
    traces = [trace for value in payloads.values() for trace in value["payload"].get("diagnostics", {}).get("mrz_crop_trace", [])]
    by_source = {trace["source_image_sha256"]: trace for trace in traces}
    fixed = json.loads((args.fixed_output / "01.fixed-crops" / "manifest.json").read_text(encoding="utf-8"))
    source_hashes = {
        doc.document_id: _source_hash(next(path for role, path in doc.paths if role == ("image" if doc.document_type == "passport" else "back")))
        for doc in documents if doc.document_type in {"passport", "id_card"}
    }
    comparisons = []
    for doc in documents:
        if doc.document_type not in {"passport", "id_card"}:
            continue
        source_hash = source_hashes[doc.document_id]
        trace = by_source.get(source_hash)
        fixed_input = next(row for row in fixed["detector_inputs"] if row["role"] == "mrz" and row["document_id"] == doc.document_id)
        fixed_lines = sorted((row for row in fixed["recognition_crops"] if row["role"] == "mrz" and row["document_id"] == doc.document_id), key=lambda row: row["key"])
        real_lines = (trace or {}).get("line_crops", [])
        reasons = []
        if trace is None:
            reasons.append("real pipeline did not emit an MRZ trace for the source image")
        else:
            if fixed_input["sha256"] != trace["normalized_crop_sha256"] or fixed_input["shape"] != trace["normalized_crop_shape"]:
                reasons.append("normalized whole-MRZ crop hash or dimensions differ")
            if len(fixed_lines) != len(real_lines):
                reasons.append(f"line count differs: fixed {len(fixed_lines)} vs real {len(real_lines)}")
            for index, (fixed_line, real_line) in enumerate(zip(fixed_lines, real_lines)):
                if fixed_line["sha256"] != real_line.get("crop_sha256") or fixed_line["shape"] != [real_line.get("original_crop_h"), real_line.get("original_crop_w"), 3]:
                    reasons.append(f"line {index + 1} crop hash or dimensions differ")
                    break
        comparisons.append({
            "document_id": doc.document_id, "document_type": doc.document_type,
            "source_image_sha256": source_hash, "fixed": {"whole": fixed_input, "lines": fixed_lines},
            "real": trace, "identical": not reasons, "reasons": reasons,
        })
    identical = all(row["identical"] for row in comparisons)
    result = {
        "identical": identical, "comparisons": comparisons,
        "root_cause": None if identical else (
            "The previous full-pipeline hook applied VOIGHT_BENCHMARK_MRZ_CROP_PREPROCESSING to the raw whole MRZ crop before the current MRZ normalization and text detection. Phase C applied the variant to detector-produced line crops after normalization. The corrected benchmark applies VOIGHT_BENCHMARK_MRZ_PREPROCESSING to those same post-detection line crops."
        ),
        "correction": "The strict rerun uses baseline MRZ normalization and baseline text detection, then applies the variant only at MRZ line recognition. The real-pipeline baseline trace is also saved as the corrected fixed-crop corpus.",
        "lifecycle": lifecycle,
    }
    write_json(output / "fixed_input_comparison.json", result)
    return result


def _run_strict(args: argparse.Namespace, documents: list[Any], output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    all_rows: list[dict[str, Any]] = []
    payloads: dict[str, Any] = {}
    for index, run in enumerate(RUN_ORDER):
        variant = "original" if run.startswith("baseline") else run
        directory = output / "runs" / f"{index:02d}_{run}"
        trace_dir = directory / "trace"
        env = {**BASE_ENV, "MODEL_DIR": str(args.model_dir),
               "VOIGHT_BENCHMARK_MRZ_PREPROCESSING": variant,
               "VOIGHT_BENCHMARK_MRZ_CROP_PREPROCESSING": "",
               "VOIGHT_BENCHMARK_MRZ_PREPROCESS_BEFORE_PACKING": "1",
               "VOIGHT_BENCHMARK_MRZ_TRACE": str(trace_dir)}
        server = Server(argparse.Namespace(port=args.port + index, model_dir=args.model_dir, timeout=args.timeout), directory, env)
        lifecycle = None
        try:
            server.start()
            for kind in DOC_TYPES:
                selected = [doc for doc in documents if doc.document_type == kind]
                warmup, _ = _post(kind, selected, args.port + index, args.timeout)
                write_json(directory / f"warmup_{kind}.json", warmup)
                for repeat in range(1, args.repeats + 1):
                    payload, elapsed = _post(kind, selected, args.port + index, args.timeout)
                    payloads[f"{run}:{kind}:{repeat}"] = payload
                    write_json(directory / f"{kind}_{repeat}.json", payload)
                    stages = _stage_totals([payload])
                    diagnostics = payload.get("diagnostics", {})
                    all_rows.append({
                        "run": run, "variant": variant, "repeat": repeat, "document_type": kind,
                        "client_e2e_seconds": elapsed, "server_e2e_seconds": payload.get("total_seconds"),
                        "mrz_preprocessing_seconds": diagnostics.get("pipeline", {}).get("mrz_crop_preprocess_seconds", 0.0),
                        "mrz_recognition_seconds": diagnostics.get("text_recognition", {}).get("elapsed_wall_seconds", 0.0),
                        "peak_rss_mb": diagnostics.get("process_peak_rss_mb"),
                        **{f"stage_{key}_seconds": value for key, value in stages.items()},
                        "correctness": _score(kind, selected, payload),
                        "documents": _request_rows(kind, selected, payload, elapsed, run, repeat),
                    })
        finally:
            lifecycle = server.stop()
            write_json(directory / "lifecycle.json", lifecycle)
        print(f"completed {run}: cleanup={lifecycle.get('cleanup_verified') if lifecycle else False}", flush=True)
    write_json(output / "strict_raw.json", all_rows)
    write_json(output / "strict_payload_index.json", {key: {"items": value.get("items", []), "diagnostics": value.get("diagnostics", {})} for key, value in payloads.items()})
    return all_rows, payloads


def _variant_name(run: str) -> str:
    return "original" if run.startswith("baseline") else run


def _aggregate(rows: list[dict[str, Any]], documents: list[Any]) -> list[dict[str, Any]]:
    output = []
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        # Correctness is deterministic here; use the first measured response and retain every raw response.
        by_doc = {item["document_id"]: item["mrz"] for row in selected for item in row["documents"] if item["mrz"]}
        passport = [value for doc, value in by_doc.items() if next(d for d in documents if d.document_id == doc).document_type == "passport"]
        ids = [value for doc, value in by_doc.items() if next(d for d in documents if d.document_id == doc).document_type == "id_card"]
        values = passport + ids
        output.append({
            "variant": variant, "passport_exact": sum(value["whole_mrz_exact"] for value in passport), "passport_total": len(passport),
            "id_exact": sum(value["whole_mrz_exact"] for value in ids), "id_total": len(ids),
            "total_exact": sum(value["whole_mrz_exact"] for value in values), "total_documents": len(values),
            "exact_lines": sum(value["exact_lines"] for value in values), "line_total": sum(value["line_total"] for value in values),
            "mrz_character_errors": sum(value["mrz_character_errors"] for value in values),
            "valid_checks": sum(all(item.get("status") != "failed" for item in value["icao_check_digit_validations"]) and bool(value["icao_check_digit_validations"]) for value in values),
        })
    return output


def _per_document_changes(rows: list[dict[str, Any]], documents: list[Any]) -> list[dict[str, Any]]:
    reference = next(row for row in rows if row["run"] == "baseline_1" and row["repeat"] == 1)
    baseline = {item["document_id"]: item["mrz"] for item in reference["documents"] if item["mrz"]}
    result = []
    first_by_variant = {}
    for row in rows:
        if row["repeat"] == 1:
            first_by_variant.setdefault(row["variant"], {}).update({item["document_id"]: item["mrz"] for item in row["documents"] if item["mrz"]})
    for document in documents:
        if document.document_type not in {"passport", "id_card"}:
            continue
        values = {variant: first_by_variant[variant][document.document_id] for variant in VARIANTS}
        notes = []
        for variant in VARIANTS[1:]:
            if values[variant]["whole_mrz_exact"] != values["original"]["whole_mrz_exact"]:
                notes.append(f"{variant} changes whole-MRZ exactness")
            elif values[variant]["mrz_character_errors"] != values["original"]["mrz_character_errors"]:
                notes.append(f"{variant} changes character errors")
        result.append({
            "document_id": document.document_id, "document_type": document.document_type,
            **{variant: {"exact": value["whole_mrz_exact"], "character_errors": value["mrz_character_errors"], "exact_lines": f"{value['exact_lines']}/{value['line_total']}"} for variant, value in values.items()},
            "best_notes": "; ".join(notes) or "unchanged",
        })
    return result


def _timing_summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        def stats(key: str, kind: str | None = None) -> str:
            values = [float(row[key]) for row in selected if kind is None or row["document_type"] == kind]
            if not values:
                return "-"
            return f"{statistics.median(values):.3f} [{min(values):.3f},{max(values):.3f}] sd={statistics.stdev(values):.3f}" if len(values) > 1 else f"{values[0]:.3f}"
        rss = [float(row["peak_rss_mb"]) for row in selected if row.get("peak_rss_mb") is not None]
        result.append({
            "variant": variant,
            "passport_e2e_seconds": stats("client_e2e_seconds", "passport"),
            "id_e2e_seconds": stats("client_e2e_seconds", "id_card"),
            "licence_e2e_control_seconds": stats("client_e2e_seconds", "driving_license"),
            "mrz_preprocessing_seconds": stats("mrz_preprocessing_seconds"),
            "mrz_recognition_stage_seconds": stats("mrz_recognition_seconds"),
            "peak_rss_mb": max(rss) if rss else None,
        })
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixed-output", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/benchmark")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/benchmarks/14.mrz-preprocessing-reconciliation")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--port", type=int, default=8041)
    args = parser.parse_args()
    if args.repeats != 5:
        parser.error("this reconciliation requires exactly five measured repeats")
    for key, value in BASE_ENV.items():
        os.environ[key] = value
    os.environ["MODEL_DIR"] = str(args.model_dir)
    documents, _ = validate_and_manifest(args.dataset_root)
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    output = args.output_root / stamp
    output.mkdir(parents=True, exist_ok=True)
    comparison = _run_preflight(args, documents, output)
    rows, _ = _run_strict(args, documents, output)
    aggregate = _aggregate(rows, documents)
    changes = _per_document_changes(rows, documents)
    timing = _timing_summary(rows)
    write_json(output / "mrz_aggregate.json", aggregate)
    write_json(output / "mrz_per_document_changes.json", changes)
    write_json(output / "timing_summary.json", timing)
    write_csv(output / "strict_raw.csv", [{key: value for key, value in row.items() if key != "documents" and key != "correctness"} for row in rows])
    write_csv(output / "mrz_aggregate.csv", aggregate)
    write_csv(output / "timing_summary.csv", timing)
    write_json(output / "run_manifest.json", {"run_order": RUN_ORDER, "variants": VARIANTS, "repeats": args.repeats, "settings": BASE_ENV, "crop_comparison": comparison["identical"]})
    print(f"completed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
