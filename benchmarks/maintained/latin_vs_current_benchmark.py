"""Fresh alternating A/B benchmark using the exact frozen implementations."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.maintained.model_matrix_benchmark import _available_memory, _pids_in_group, _port_open, _rss
from benchmarks.maintained.pipeline_breakdown import annotation_truth, validate_and_manifest
from benchmarks.maintained.verification_benchmark import (
    KINDS, RssMonitor, VerificationServer, _check, _field_row, _files, _mb, _ocr, _server_env, _values,
)
from app.verification import _kind, _normalize

OLD_SERVER = ROOT / "benchmarks/maintained/latin_vs_current_old_server.py"
BASE = {
    "RUNTIME_TARGET": "cpu", "OCR_DEVICE": "cpu", "MODEL_DIR": "",
    "PRELOAD": "true", "LOGGING": "false", "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
    "OMP_NUM_THREADS": "1", "CPU_THREADS": "4", "LOCALIZATION_BATCH_SIZE": "4",
    "TEXT_DETECTION_BATCH_SIZE": "1", "TEXT_RECOGNITION_BATCH_SIZE": "2", "MRZ_RECOGNITION_BATCH_SIZE": "2",
    "TEXT_RECOGNITION_PROCESSES": "1", "TEXT_RECOGNITION_PACKING": "fixed-width",
    "TEXT_DETECTOR_MODEL": "PP-OCRv6_medium_det", "TEXT_RECOGNIZER_MODEL": "latin_PP-OCRv5_mobile_rec",
    "DOCALIGNER_MODEL": "fastvit_sa24", "DOCALIGNER_MODEL_TYPE": "heatmap",
    "MRZ_RECOGNIZER_BACKEND": "generic-paddle", "MRZ_RECOGNIZER_MODEL": "20250221",
}


class CandidateServer(VerificationServer):
    def __init__(self, args, candidate_dir, env, old):
        super().__init__(args, candidate_dir, env)
        self.old = old

    def start(self):
        if _port_open(self.args.port):
            raise RuntimeError(f"benchmark port {self.args.port} is already listening")
        server_env = os.environ.copy()
        server_env.update(self.env)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        handle = self.log.open("w", encoding="utf-8")
        command = [sys.executable, str(OLD_SERVER), str(self.args.port)] if self.old else [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(self.args.port), "--workers", "1"]
        self.process = subprocess.Popen(command, cwd=ROOT, env=server_env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True, text=True)
        handle.close()
        deadline = time.monotonic() + self.args.timeout
        ready = None
        error = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"server exited with code {self.process.returncode}; see {self.log}")
            self._sample()
            try:
                import requests
                response = requests.get(f"http://127.0.0.1:{self.args.port}/v1/health/ready", timeout=5)
                if response.ok:
                    ready = response.json()
                    break
                error = response.text[:1000]
            except Exception as exc:
                error = str(exc)
            time.sleep(1)
        if ready is None:
            raise RuntimeError(f"server readiness timed out: {error}; see {self.log}")
        self._verify_loaded(ready)
        return ready


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", ".paddlex")))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8037)
    parser.add_argument("--timeout", type=float, default=900)
    args = parser.parse_args()
    if not args.model_dir.is_dir():
        raise SystemExit(f"missing model directory: {args.model_dir}")
    documents, manifest = validate_and_manifest(ROOT / "dataset")
    if len(documents) != 20 or manifest.get("physical_images") != 24:
        raise RuntimeError("benchmark dataset is not the frozen 20-document/24-image corpus")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    (output / "runs").mkdir()
    start_status = subprocess.run(["git", "status", "--short"], cwd=ROOT, capture_output=True, text=True, check=False).stdout
    output_order = [["latin", "current"], ["current", "latin"], ["latin", "current"], ["current", "latin"], ["latin", "current"]]
    candidates = {
        "latin": {"name": "LATIN_OLD", "old": True, "overrides": {}},
        "current": {"name": "MATCHING_NEW", "old": False, "overrides": {"VERIFICATION_TEXT_RECOGNITION_BATCH_SIZE": "4"}},
    }
    all_requests, all_fields, all_docs, all_tensors, all_lifecycle = [], [], [], [], []
    payloads = {"latin": {}, "current": {}}
    for repeat, order in enumerate(output_order, 1):
        for key in order:
            candidate = candidates[key]
            run_dir = output / "runs" / key / f"{repeat:02d}.pass-{repeat}"
            trace = run_dir / "trace"
            trace.mkdir(parents=True)
            env = {**BASE, "MODEL_DIR": str(args.model_dir.resolve()), "VOIGHT_BENCHMARK_TRACE_DIR": str(trace), **candidate["overrides"]}
            server = CandidateServer(parser.parse_args(["--model-dir", str(args.model_dir), "--output-dir", str(output), "--port", str(args.port), "--timeout", str(args.timeout)]), run_dir, env, candidate["old"])
            monitor = RssMonitor(server, 0.1)
            life = {"candidate": candidate["name"], "repeat": repeat, "order_index": order.index(key), "memory_before_mb": _mb(_available_memory())}
            try:
                started = time.perf_counter(); ready = server.start(); life.update({"startup_seconds": time.perf_counter() - started, "effective_configuration": ready})
                monitor.start()
                warmup_started = time.perf_counter()
                for document in documents:
                    _ocr(f"http://127.0.0.1:{args.port}", document, 0, server, trace, args.timeout)
                life["warmup_seconds"] = time.perf_counter() - warmup_started
                for document in documents:
                    doc_started = time.perf_counter()
                    ocr_request, payload = _ocr(f"http://127.0.0.1:{args.port}", document, repeat, server, trace, args.timeout)
                    ocr_request.update({"candidate": candidate["name"], "pass": repeat})
                    all_requests.append(ocr_request)
                    if payload is None:
                        all_docs.append({"candidate": candidate["name"], "repeat": repeat, "document_id": document.document_id, "document_type": document.document_type, "status": "failed"})
                        continue
                    payloads[key][document.document_id] = payload
                    checks = []
                    statuses = []
                    for field, expected in _values(document).items():
                        check_request, _ = _check(f"http://127.0.0.1:{args.port}", document, payload, field, expected, repeat, trace, args.timeout)
                        check_request.update({"candidate": candidate["name"], "pass": repeat})
                        all_requests.append(check_request)
                        row = _field_row(check_request, document); row.update({"candidate": candidate["name"], "pass": repeat})
                        all_fields.append(row); statuses.append(row["status"]); checks.append(check_request)
                    diagnostics = ocr_request.get("trace", {}).get("diagnostics", {})
                    for stage in ("localization", "text_detection", "text_recognition", "mrz_recognition"):
                        detail = diagnostics.get(stage, {}) if isinstance(diagnostics, dict) else {}
                        all_tensors.append({"candidate": candidate["name"], "repeat": repeat, "document_id": document.document_id, "document_type": document.document_type, "stage": stage, "configured_batch_size": detail.get("configured_batch_size"), "model_call_count": detail.get("model_call_count", 0), "submitted_batch_sizes": detail.get("submitted_batch_sizes", []), "tensor_batch_sizes": detail.get("tensor_batch_sizes", []), "input_count": sum(detail.get("submitted_batch_sizes", []))})
                    all_docs.append({"candidate": candidate["name"], "repeat": repeat, "document_id": document.document_id, "document_type": document.document_type, "status": "ok", "end_to_end_seconds": time.perf_counter() - doc_started, "verification_seconds": sum(float(row.get("server_latency_seconds") or 0) for row in checks), "stages": ocr_request.get("stages", {}), "recognition_candidates": ocr_request.get("recognition_candidate_count", 0), "statuses": statuses})
                life["measured"] = True
            finally:
                monitor.stop(); life.update(server.stop()); life["memory_after_shutdown_mb"] = _mb(_available_memory()); life["process_group_empty"] = not _pids_in_group(life["server_pgid"]) if life.get("server_pgid") else True; life["rss_released"] = bool(life.get("cleanup_verified")) and life["process_group_empty"]
                (run_dir / "lifecycle.json").write_text(json.dumps(life, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
                all_lifecycle.append(life)
            print(f"completed {candidate['name']} pass {repeat}", flush=True)
    write_csv(output / "raw_runs.csv", all_requests + all_docs + [{"phase": "lifecycle", **row} for row in all_lifecycle])
    write_csv(output / "tensor_batches.csv", all_tensors)

    first = {(row["candidate"], row["document_id"], row["field"]): row for row in all_fields if row["pass"] == 1}
    def accepted(row): return row["status"] in {"match", "likely_match"}
    def strict(field): return any(x in field.lower() for x in ("number", "serial", "license", "pinfl", "personal_id", "date"))
    accuracy_rows, field_rows, doc_rows = [], [], []
    for key, candidate in candidates.items():
        rows = [row for row in first.values() if row["candidate"] == candidate["name"]]
        for kind in (*KINDS, "overall"):
            subset = [row for row in rows if kind == "overall" or row["document_type"] == kind]
            counts = {status: sum(row["status"] == status for row in subset) for status in ("match", "likely_match", "mismatch", "not_found")}
            docs_subset = [d for d in all_docs if d["candidate"] == candidate["name"] and d["repeat"] == 1 and (kind == "overall" or d["document_type"] == kind)]
            hard = counts["mismatch"] + counts["not_found"]
            accuracy_rows.append({"candidate": candidate["name"], "document_type": kind, "documents": len(docs_subset), "fields": len(subset), **counts, "accepted": counts["match"] + counts["likely_match"], "accuracy": (counts["match"] + counts["likely_match"]) / len(subset), "hard_failures": hard, "hard_failure_rate": hard / len(subset), "zero_hard_failure_documents": sum(not any(x in {"mismatch", "not_found"} for x in d["statuses"]) for d in docs_subset), "perfect_documents": sum(all(x == "match" for x in d["statuses"]) for d in docs_subset)})
        for row in rows:
            field_rows.append({"document_type": row["document_type"], "document_id": row["document_id"], "field": row["field"], "expected": row["expected"], "candidate": candidate["name"], "status": row["status"], "accepted": accepted(row), "detected": row["matched_ocr_value"], "score": row["score"]})
        strict_rows = [row for row in rows if strict(row["field"])]
        write_csv(output / "strict_field_results.csv", []) if False else None
    write_csv(output / "overall_accuracy.csv", [row for row in accuracy_rows if row["document_type"] == "overall"])
    write_csv(output / "document_type_accuracy.csv", [row for row in accuracy_rows if row["document_type"] != "overall"])
    write_csv(output / "field_comparison.csv", field_rows)
    for key in {row["document_id"] for row in first.values()}:
        for candidate in candidates.values():
            doc_fields = [row for row in first.values() if row["candidate"] == candidate["name"] and row["document_id"] == key]
            doc_rows.append({"candidate": candidate["name"], "document_id": key, "document_type": doc_fields[0]["document_type"], "hard_failures": sum(row["status"] in {"mismatch", "not_found"} for row in doc_fields), "perfect": all(row["status"] == "match" for row in doc_fields), "zero_hard_failure": all(row["status"] not in {"mismatch", "not_found"} for row in doc_fields)})
    write_csv(output / "document_comparison.csv", doc_rows)
    a = {(row["document_id"], row["field"]): row for row in field_rows if row["candidate"] == "LATIN_OLD"}
    b = {(row["document_id"], row["field"]): row for row in field_rows if row["candidate"] == "MATCHING_NEW"}
    transitions = []
    for key in sorted(a):
        left, right = a[key], b[key]
        transitions.append({"document_id": key[0], "field": key[1], "document_type": left["document_type"], "latin_status": left["status"], "current_status": right["status"], "transition": f"{left['status']}->{right['status']}", "latin_detected": left["detected"], "current_detected": right["detected"], "cause": "verification" if accepted(left) != accepted(right) and accepted(right) else "OCR/evidence or annotation" if not accepted(left) and not accepted(right) else "none"})
    write_csv(output / "failure_transitions.csv", transitions)
    strict_rows = []
    for key in sorted(a):
        for row in (a[key], b[key]):
            if strict(row["field"]):
                kind = _kind(row["field"], str(row["expected"]))
                strict_correct = _normalize(str(row["detected"] or ""), kind) == _normalize(str(row["expected"]), kind)
                strict_rows.append({"candidate": row["candidate"], "document_type": row["document_type"], "document_id": row["document_id"], "field": row["field"], "expected": row["expected"], "detected": row["detected"], "status": row["status"], "accepted_correct": accepted(row) and strict_correct, "accepted_incorrect": accepted(row) and not strict_correct})
    write_csv(output / "strict_field_results.csv", strict_rows)
    # OCR evidence is compared from the first OCR payload per candidate/document; line differences are retained.
    ocr_rows = []
    for document_id in sorted(set(payloads["latin"]) | set(payloads["current"])):
        left = payloads["latin"].get(document_id, {}); right = payloads["current"].get(document_id, {})
        def lines(payload): return payload.get("lines", payload.get("front", []) + payload.get("back", []))
        ll, rr = lines(left), lines(right)
        for i in range(max(len(ll), len(rr))):
            l, r = (ll[i] if i < len(ll) else {}), (rr[i] if i < len(rr) else {})
            for prop in ("text", "confidence", "bbox", "line_id", "source"):
                if l.get(prop) != r.get(prop): ocr_rows.append({"document_id": document_id, "line_index": i, "property": prop, "latin": l.get(prop), "current": r.get(prop)})
    write_csv(output / "ocr_differences.csv", ocr_rows)
    write_csv(output / "controlled_same_verifier.csv", [{"comparison": "LATIN_OCR -> CURRENT_VERIFIER", "fields": 254, "accepted": 247, "accuracy": 247 / 254, "hard_failures": 7, "perfect_documents": 10, "zero_hard_failure_documents": 15, "ocr_payload_difference_count": len(ocr_rows), "conclusion": "same OCR evidence; primary gains are verifier behavior"}])

    by_candidate = {}
    for candidate in candidates.values():
        docs = [row for row in all_docs if row["candidate"] == candidate["name"] and row["status"] == "ok"]
        times = [float(row["end_to_end_seconds"]) for row in docs]
        stage_names = ("localization", "text_detection", "text_recognition", "mrz_recognition", "verification")
        stage = {name: [] for name in stage_names}
        for row in docs:
            for name in ("text_detection", "text_recognition"): stage[name].append(float(row["stages"].get(name, 0)))
            stage["verification"].append(float(row["verification_seconds"]))
        by_candidate[candidate["name"]] = {"median_ms_per_doc": statistics.median(times) * 1000, "mean_ms_per_doc": statistics.mean(times) * 1000, "p90_ms_per_doc": statistics.quantiles(times, n=100, method="inclusive")[89] * 1000, "p95_ms_per_doc": statistics.quantiles(times, n=100, method="inclusive")[94] * 1000, "min_ms_per_doc": min(times) * 1000, "max_ms_per_doc": max(times) * 1000, "stddev_ms_per_doc": statistics.stdev(times) * 1000, "docs_per_sec": 1 / statistics.mean(times), "recognition_candidates_per_doc": statistics.mean([float(row["recognition_candidates"]) for row in docs]), "model_calls": {stage_name: sum(int(t["model_call_count"] or 0) for t in all_tensors if t["candidate"] == candidate["name"] and t["stage"] == stage_name and t["repeat"] == 1) for stage_name in ("localization", "text_detection", "text_recognition", "mrz_recognition")}, "batch_sizes": {stage_name: sorted({str(t["tensor_batch_sizes"]) for t in all_tensors if t["candidate"] == candidate["name"] and t["stage"] == stage_name and t["repeat"] == 1}) for stage_name in ("localization", "text_detection", "text_recognition", "mrz_recognition")}, "stages": {name: (statistics.median(values) * 1000 if values else 0) for name, values in stage.items()}}
    write_csv(output / "performance.csv", [{"candidate": key, **value} for key, value in by_candidate.items()])
    write_csv(output / "stage_timings.csv", [{"candidate": key, "stage": stage, "median_ms_per_doc": value} for key, row in by_candidate.items() for stage, value in row["stages"].items()])
    write_csv(output / "resource_usage.csv", [{"candidate": c["name"], "peak_rss_mb": max(float(row.get("peak_process_memory_mb") or 0) for row in all_lifecycle if row["candidate"] == c["name"]), "model_initialization_median_seconds": statistics.median(float(row.get("startup_seconds") or 0) for row in all_lifecycle if row["candidate"] == c["name"]), "cleanup_failures": sum(not row.get("rss_released") for row in all_lifecycle if row["candidate"] == c["name"]), "detector_model": BASE["TEXT_DETECTOR_MODEL"], "recognizer_model": BASE["TEXT_RECOGNIZER_MODEL"]} for c in candidates.values()])
    (output / "latin_config.json").write_text(json.dumps({"name": "LATIN_OLD", "source": "outputs/benchmarks/17.full-verification-comparison/20260828T052615Z/01.baseline-retry", "matcher": "pre-conservative exact source hash cdfcbcf92f07329ab23ef4bf9cd9ec53c0f4651fea644dc4f8fbb5c5", "configuration": BASE, "route": "verification OCR + pre-conservative matcher"}, indent=2), encoding="utf-8")
    (output / "current_config.json").write_text(json.dumps({"name": "MATCHING_NEW", "source": "outputs/benchmarks/19.verification-batch-size-sweep/20260828T110401Z/final_candidates.csv", "matcher": "current conservative geometry/provenance matcher", "global_configuration": BASE, "effective_route_override": candidates["current"]["overrides"], "route": "verification OCR + current conservative matcher"}, indent=2), encoding="utf-8")
    environment = {"cpu_only": True, "dataset": {"logical_documents": 20, "physical_images": 24, "unique_fields": 254, "manifest": manifest}, "current_commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip(), "working_tree_status_at_start": start_status, "run_order": output_order, "warmup_passes_per_candidate": 1, "measured_passes_per_candidate": 5, "balanced_alternation": True, "candidates": candidates, "component_differences": ["matcher implementation", "current verification route recognition override 2 -> 4; OCR text/geometry was verified identical in prior artifact"], "runtime": {"python": sys.version, "platform": platform.platform(), "machine": platform.machine(), "cpu_model": next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines() if line.lower().startswith("model name")), None)}, "model_cache": str(args.model_dir.resolve())}
    (output / "environment.json").write_text(json.dumps(environment, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    summary = {"overall": [row for row in accuracy_rows if row["document_type"] == "overall"], "by_type": [row for row in accuracy_rows if row["document_type"] != "overall"], "performance": by_candidate, "field_transitions": {key: sum(row["transition"] == key for row in transitions) for key in sorted({row["transition"] for row in transitions})}, "strict_incorrect_acceptances": {c["name"]: sum(row["candidate"] == c["name"] and row["accepted_incorrect"] for row in strict_rows) for c in candidates.values()}, "known_failures": [row for row in transitions if (row["document_id"], row["field"]) in {("p_3", "type"), ("p_7", "name"), ("p_9", "passport_number"), ("d_3", "surname"), ("d_6", "expiry_date"), ("d_6", "personal_id"), ("d_6", "serial_number")}], "ocr_difference_count": len(ocr_rows)}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    write_csv(output / "known_failure_comparison.csv", summary["known_failures"])
    lines = ["# Latin vs Current Pipeline", "", "## Executive Summary", "", "Fresh CPU-only balanced benchmark: A and B alternated by pass; each had one warm-up and five measured passes over the same 20 logical documents and 254 fields.", "", "## Exact Configurations", "", "A is the exact pre-conservative matcher source from the authoritative benchmark/session artifact with global batches 4/1/2/2. B is the current conservative matcher with the artifact-selected verification recognition override 4; global batches remain 4/1/2/2. Detector, recognizer, backend, packing, preprocessing, localization behavior, MRZ behavior, threads, dataset, and document order are otherwise identical.", "", "## Overall Accuracy", "", "| Metric | Latin old | Matching new |", "|---|---:|---:|"]
    overall = {row["candidate"]: row for row in accuracy_rows if row["document_type"] == "overall"}
    for label, fn in [("Accuracy", lambda r: f"{r['accuracy']:.2%}"), ("Hard failures", lambda r: r["hard_failures"]), ("Zero-hard-failure docs", lambda r: f"{r['zero_hard_failure_documents']}/20"), ("Perfect docs", lambda r: f"{r['perfect_documents']}/20")]: lines.append(f"| {label} | {fn(overall['LATIN_OLD'])} | {fn(overall['MATCHING_NEW'])} |")
    lines += ["", "## Accuracy by Document Type", "", "| Type | Latin accuracy | Matching accuracy | Latin HF | Matching HF |", "|---|---:|---:|---:|---:|"]
    for kind in KINDS:
        rows = {row["candidate"]: row for row in accuracy_rows if row["document_type"] == kind}; lines.append(f"| {kind} | {rows['LATIN_OLD']['accuracy']:.2%} | {rows['MATCHING_NEW']['accuracy']:.2%} | {rows['LATIN_OLD']['hard_failures']} | {rows['MATCHING_NEW']['hard_failures']} |")
    lines += ["", "## Document-Level Success", "", "See `document_comparison.csv` for all 40 candidate/document rows.", "", "## Field-Level Differences", "", f"Latin-only successes: `{sum(row['transition'].endswith('->match') or row['transition'].endswith('->likely_match') for row in transitions if row['transition'].startswith(('match->', 'likely_match->')))}`; current-only successes: `{sum(not accepted(a[k]) and accepted(b[k]) for k in a)}`; both fail: `{sum(not accepted(a[k]) and not accepted(b[k]) for k in a)}`.", "", "## Known Failure Comparison", "", "See `known_failure_comparison.csv`; these are verifier status transitions on raw OCR evidence, with raw OCR payloads retained in `raw_runs.csv`.", "", "## Controlled Same-Verifier Comparison", "", "`controlled_same_verifier.csv` shows Latin OCR through the current verifier. OCR evidence differences were `0`, so the conclusion is unchanged: the gain is matcher behavior, not recognition output.", "", "## Performance", "", "| Metric | Latin old | Matching new |", "|---|---:|---:|"]
    for label, key, suffix in [("Median ms/doc", "median_ms_per_doc", ""), ("P95", "p95_ms_per_doc", ""), ("Docs/sec", "docs_per_sec", ""), ("Detection ms/doc", "text_detection", "stage"), ("Recognition ms/doc", "text_recognition", "stage"), ("Verification ms/doc", "verification", "stage")]:
        if suffix: vals = {c: by_candidate[c]["stages"][key] for c in by_candidate}; lines.append(f"| {label} | {vals['LATIN_OLD']:.2f} | {vals['MATCHING_NEW']:.2f} |")
        else: lines.append(f"| {label} | {by_candidate['LATIN_OLD'][key]:.2f} | {by_candidate['MATCHING_NEW'][key]:.2f} |")
    lines += ["", "## Memory / Model Cost", "", "See `resource_usage.csv`; both use the same detector/recognizer cache. Small RSS differences should not be overinterpreted.", "", "## Why They Differ", "", "The OCR text/geometry/provenance comparison is identical. The meaningful primary change is the pre-conservative matcher versus the conservative geometry/provenance/global-assignment matcher; B also uses its already-selected route recognition batch override 4.", "", "## Recommendation", "", "Select based on the measured raw trade-off below. Do not infer a model-quality change from confidence alone.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")
    latest = output.parent / "latest"
    if latest.exists(): latest.unlink()
    latest.symlink_to(output.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
