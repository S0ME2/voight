"""Profile API time outside the three top-level OCR model stages."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.maintained.full_dataset_api_batch_benchmark import (  # noqa: E402
    KINDS,
    _payload,
    _records,
)

ROUTES = {
    "full-latin-pipeline": "full Latin pipeline (/v1/ocr/{type}/batch)",
    "comparison": "comparison routes (/verification/{type}/ocr/batch + /check)",
}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 7, 8, 9, 16])
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--route", choices=tuple(ROUTES), default="full-latin-pipeline")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 3 or any(size < 1 for size in args.sizes):
        parser.error("--repeats must be at least 3 and --sizes must be positive")
    return args


def _post_full(base_url: str, kind: str, records: list[dict[str, Any]], count: int, timeout: float) -> dict[str, Any]:
    files, manifest = _payload(kind, records, count)
    started = time.perf_counter()
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/v1/ocr/{kind}/batch",
            files=files,
            timeout=timeout,
        )
        client_seconds = time.perf_counter() - started
        payload = response.json()
    except Exception as error:
        return {
            "status": "request_failed",
            "failure_detail": f"{type(error).__name__}: {error}",
            "document_type": kind,
            "document_count": count,
            **manifest,
    }
    diagnostics = payload.get("diagnostics", {})
    if not diagnostics.get("benchmark_profile"):
        raise RuntimeError("full-latin-pipeline profiling is disabled; restart the service with VOIGHT_BENCHMARK_PROFILE=true")
    return {
        "status": "ok" if response.ok and payload.get("failed", 0) == 0 else "request_failed",
        "document_type": kind,
        "document_count": count,
        "client_wall_seconds": client_seconds,
        "server_wall_seconds": payload.get("total_seconds"),
        "documents_per_second": count / client_seconds if client_seconds else None,
        "succeeded": payload.get("succeeded"),
        "failures": payload.get("failed"),
        "failure_detail": None if response.ok else response.text[:1000],
        "profile_ns": diagnostics.get("benchmark_profile", {}),
        "pipeline": diagnostics.get("pipeline", {}),
        "localization": diagnostics.get("localization", {}),
        "preprocessing": diagnostics.get("preprocessing_seconds", {}),
        "line_crop_seconds": diagnostics.get("line_crop_seconds", 0.0),
        "result_unpack_seconds": diagnostics.get("result_unpack_seconds", 0.0),
        "text_detection": diagnostics.get("text_detection", {}),
        "text_recognition": diagnostics.get("text_recognition", {}),
        "mrz_recognition": diagnostics.get("mrz_recognition", {}),
        **manifest,
    }


def _comparison_fields(record: dict[str, Any]) -> dict[str, str]:
    return {
        name: str(entry["value"])
        for name, entry in record["fields"].items()
        if entry.get("state") == "value" and entry.get("value") not in (None, "")
    }


def _post_comparison(base_url: str, kind: str, records: list[dict[str, Any]], count: int, timeout: float) -> dict[str, Any]:
    files, manifest = _payload(kind, records, count)
    verification_kind = "driving-licence" if kind == "driving-license" else kind
    started = time.perf_counter()
    try:
        response = requests.post(
            f"{base_url.rstrip('/')}/verification/{verification_kind}/ocr/batch",
            files=files,
            timeout=timeout,
        )
        ocr_client_seconds = time.perf_counter() - started
        ocr_payload = response.json()
        if not response.ok:
            return {"status": "request_failed", "document_type": kind, "document_count": count, "failure_detail": response.text[:1000], "ocr_client_wall_seconds": ocr_client_seconds, **manifest}
    except Exception as error:
        return {"status": "request_failed", "document_type": kind, "document_count": count, "failure_detail": f"{type(error).__name__}: {error}", **manifest}

    checks_started = time.perf_counter()
    check_failures = []
    summary = {"match": 0, "likely_match": 0, "mismatch": 0, "not_found": 0}
    for index, item in enumerate(ocr_payload.get("items", [])):
        if not item.get("success"):
            check_failures.append({"index": index, "error": item.get("error")})
            continue
        record = records[index % len(records)]
        try:
            check = requests.post(
                f"{base_url.rstrip('/')}/verification/{verification_kind}/check",
                json={"ocr": item["result"], "fields": _comparison_fields(record)},
                timeout=timeout,
            )
            if not check.ok:
                check_failures.append({"index": index, "error": check.text[:1000]})
                continue
            check_payload = check.json()
            for key in summary:
                summary[key] += int(check_payload.get("summary", {}).get(key, 0))
        except Exception as error:
            check_failures.append({"index": index, "error": f"{type(error).__name__}: {error}"})
    check_client_seconds = time.perf_counter() - checks_started
    client_seconds = time.perf_counter() - started
    return {
        "status": "ok" if not check_failures and ocr_payload.get("failed", 0) == 0 else "request_failed",
        "document_type": kind,
        "document_count": count,
        "client_wall_seconds": client_seconds,
        "ocr_client_wall_seconds": ocr_client_seconds,
        "check_client_wall_seconds": check_client_seconds,
        "documents_per_second": count / client_seconds if client_seconds else None,
        "succeeded": ocr_payload.get("succeeded"),
        "failures": ocr_payload.get("failed", 0) + len(check_failures),
        "check_failures": check_failures,
        "comparison_summary": summary,
        **manifest,
    }


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def _profile_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            groups.setdefault((row["document_type"], row["document_count"]), []).append(row)
    output = []
    for (kind, count), values in sorted(groups.items()):
        keys = sorted({key for row in values for key in row.get("profile_ns", {}) if key.endswith("_ns")})
        server_seconds = _median([float(row["server_wall_seconds"]) for row in values])
        for key in keys:
            value = _median([float(row["profile_ns"].get(key, 0)) for row in values])
            output.append({
                "document_type": kind,
                "document_count": count,
                "metric": key,
                "median_ns": value,
                "median_ms": value / 1_000_000 if value is not None else None,
                "percent_of_server_wall": (value / (server_seconds * 1_000_000_000) * 100) if value is not None and server_seconds else None,
            })
    return output


def _stage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create a non-overlapping-ish top-level view for easy comparison.

    The detailed profile remains authoritative. These values are deliberately
    labelled as boundaries: model stages and artifact writes are nested.
    """
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            groups.setdefault((row["document_type"], row["document_count"]), []).append(row)
    output = []
    for (kind, count), values in sorted(groups.items()):
        server = _median([float(row["server_wall_seconds"]) for row in values]) or 0.0
        for label, value in (
            ("upload_read", sum(float(row["profile_ns"].get("api.upload_read_ns", 0)) for row in values) / len(values)),
            ("archive_parse", sum(float(row["profile_ns"].get("api.archive_parse_ns", 0)) for row in values) / len(values)),
            ("plan", sum(float(row["profile_ns"].get(key, 0)) for row in values for key in ("plan.image_decode_ns", "plan.profile_load_ns", "plan.input_artifact_ns")) / len(values)),
            ("localization_orchestration", sum(float(row["profile_ns"].get("runner.localization_orchestration_ns", 0)) for row in values) / len(values)),
            ("document_preparation", sum(float(row["profile_ns"].get("runner.document_preparation_ns", 0)) for row in values) / len(values)),
            ("mrz_crop_preparation", sum(float(row["profile_ns"].get("runner.mrz_crop_preparation_ns", 0)) for row in values) / len(values)),
            ("ocr_pipeline", sum(float(row["profile_ns"].get("runner.ocr_pipeline_ns", 0)) for row in values) / len(values)),
            ("mrz_recognition", sum(float(row["profile_ns"].get("runner.mrz_recognition_ns", 0)) for row in values) / len(values)),
            ("result_assembly", sum(float(row["profile_ns"].get("runner.result_assembly_ns", 0)) for row in values) / len(values)),
            ("response_assembly", sum(float(row["profile_ns"].get("api.response_assembly_ns", 0)) for row in values) / len(values)),
            ("endpoint_artifact_output", sum(float(row["profile_ns"].get(key, 0)) for row in values for key in ("api.pipeline_diagnostics_save_ns", "api.response_artifact_save_ns")) / len(values)),
            ("detector_model", sum(float(row.get("text_detection", {}).get("wall_seconds", 0.0)) for row in values) * 1_000_000_000 / len(values)),
            ("recognizer_model", sum(float(row.get("text_recognition", {}).get("wall_seconds", 0.0)) for row in values) * 1_000_000_000 / len(values)),
            ("line_crop", sum(float(row.get("line_crop_seconds", 0.0)) for row in values) * 1_000_000_000 / len(values)),
            ("detector_preprocessing", sum(float(row.get("preprocessing", {}).get("detector", 0.0)) for row in values) * 1_000_000_000 / len(values)),
            ("visible_preprocessing", sum(float(row.get("preprocessing", {}).get("visible", 0.0)) for row in values) * 1_000_000_000 / len(values)),
            ("mrz_preprocessing", sum(float(row.get("preprocessing", {}).get("mrz", 0.0)) for row in values) * 1_000_000_000 / len(values)),
            ("result_unpack", sum(float(row.get("result_unpack_seconds", 0.0)) for row in values) * 1_000_000_000 / len(values)),
            ("ocr_wrapper_residual", sum(
                max(0.0, float(value["profile_ns"].get("runner.ocr_pipeline_ns", 0))
                - float(value.get("text_detection", {}).get("wall_seconds", 0.0)) * 1_000_000_000
                - float(value.get("text_recognition", {}).get("wall_seconds", 0.0)) * 1_000_000_000
                - float(value.get("line_crop_seconds", 0.0)) * 1_000_000_000
                - sum(float(value.get("preprocessing", {}).get(name, 0.0)) for name in ("detector", "visible", "mrz")) * 1_000_000_000
                - float(value.get("result_unpack_seconds", 0.0)) * 1_000_000_000
            ) for value in values) / len(values)),
        ):
            output.append({
                "document_type": kind,
                "document_count": count,
                "stage": label,
                "median_ms": value / 1_000_000,
                "percent_of_server_wall": value / (server * 1_000_000_000) * 100 if server else None,
            })
    return output


def _comparison_stage_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("status") == "ok":
            groups.setdefault((row["document_type"], row["document_count"]), []).append(row)
    output = []
    for (kind, count), values in sorted(groups.items()):
        total = _median([float(row["client_wall_seconds"]) for row in values]) or 0.0
        for stage, key in (("ocr_route", "ocr_client_wall_seconds"), ("check_routes", "check_client_wall_seconds"), ("total_client", "client_wall_seconds")):
            value = _median([float(row.get(key, 0.0)) for row in values]) or 0.0
            output.append({
                "document_type": kind,
                "document_count": count,
                "stage": stage,
                "median_ms": value * 1000,
                "percent_of_server_wall": value / total * 100 if total else None,
            })
    return output


def main() -> None:
    args = _args()
    records = {kind: _records(args.dataset_root, kind) for kind in KINDS}
    post = _post_full if args.route == "full-latin-pipeline" else _post_comparison
    warmups = [post(args.base_url, kind, values, len(values), args.timeout) for kind, values in records.items()]
    rows = []
    for size in args.sizes:
        for kind in KINDS:
            rows.extend(post(args.base_url, kind, records[kind], size, args.timeout) for _ in range(args.repeats))
    report = {
        "method": ROUTES[args.route],
        "route": args.route,
        "profile_clock": "perf_counter_ns",
        "dataset": {
            kind: {
                "logical_documents": len(values),
                "physical_images": sum(len(record["images"]) for record in values),
                "ids": [record["id"] for record in values],
            }
            for kind, values in records.items()
        },
        "sizes": args.sizes,
        "repeats": args.repeats,
        "warmup": warmups,
        "rows": rows,
        "profile_rows": _profile_rows(rows) if args.route == "full-latin-pipeline" else [],
        "stage_rows": _stage_rows(rows) if args.route == "full-latin-pipeline" else _comparison_stage_rows(rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as output:
        fields = ("document_type", "document_count", "metric", "median_ns", "median_ms", "percent_of_server_wall")
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["profile_rows"])
    with args.output.with_name("stage_breakdown.csv").open("w", newline="", encoding="utf-8") as output:
        fields = ("document_type", "document_count", "stage", "median_ms", "percent_of_server_wall")
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(report["stage_rows"])
    print(args.output)


if __name__ == "__main__":
    main()
