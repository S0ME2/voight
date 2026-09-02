"""Benchmark the real whole-image verification OCR and check routes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.maintained.model_matrix_benchmark import (
    Server,
    _available_memory,
    _pids_in_group,
    _rss,
)
from benchmarks.maintained.pipeline_breakdown import (
    DOC_TYPES,
    Document,
    annotation_truth,
    validate_and_manifest,
)

KINDS = ("passport", "id_card", "driving_license")
ROUTES = {
    "passport": ("passport", "/verification/passport/ocr", "/verification/passport/check"),
    "id_card": ("id-card", "/verification/id-card/ocr", "/verification/id-card/check"),
    "driving_license": ("driving-licence", "/verification/driving-licence/ocr", "/verification/driving-licence/check"),
}
MODEL_ENV = {
    "TEXT_DETECTOR_MODEL": "PP-OCRv6_medium_det",
    "TEXT_RECOGNIZER_MODEL": "latin_PP-OCRv5_mobile_rec",
    "DOCALIGNER_MODEL": "fastvit_sa24",
    "DOCALIGNER_MODEL_TYPE": "heatmap",
    "MRZ_RECOGNIZER_BACKEND": "generic-paddle",
    "MRZ_RECOGNIZER_MODEL": "20250221",
}


class VerificationServer(Server):
    """Use the shared lifecycle but validate only readiness model sections."""

    def _verify_loaded(self, ready: dict[str, Any]) -> None:
        models = ready.get("models", {})
        expected = {
            "text_detector": {"model": self.env["TEXT_DETECTOR_MODEL"]},
            "text_recognizer": {"model": self.env["TEXT_RECOGNIZER_MODEL"]},
            "document_localizer": {"model_cfg": self.env["DOCALIGNER_MODEL"]},
            "mrz_localizer": {"model_cfg": "20250222"},
            "mrz_recognizer": {"backend": self.env["MRZ_RECOGNIZER_BACKEND"]},
        }
        for section, checks in expected.items():
            actual = models.get(section, {})
            if not actual.get("loaded"):
                raise RuntimeError(f"model section did not report loaded: {section}: {actual}")
            for key, value in checks.items():
                if actual.get(key) != value:
                    raise RuntimeError(f"loaded configuration mismatch for {section}.{key}: expected {value!r}, got {actual.get(key)!r}")


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", ".paddlex")))
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/benchmarks/16.verification-baseline")
    parser.add_argument("--output-dir", type=Path, help="Use this exact run directory instead of a UTC timestamp")
    parser.add_argument("--document-type", choices=("all", *KINDS), default="all")
    parser.add_argument("--limit", type=int, help="Keep the first N documents of each selected type")
    parser.add_argument("--port", type=int, default=8013)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--rss-interval", type=float, default=0.1)
    parsed = parser.parse_args()
    if parsed.repeats < 3 or parsed.warmup != 1:
        parser.error("use exactly one warm-up and at least three measured repeats")
    if parsed.limit is not None and parsed.limit < 1:
        parser.error("--limit must be positive")
    if parsed.timeout <= 0 or parsed.rss_interval <= 0:
        parser.error("--timeout and --rss-interval must be positive")
    if not parsed.model_dir.is_dir():
        parser.error(f"model directory does not exist: {parsed.model_dir}")
    parsed.kinds = KINDS if parsed.document_type == "all" else (parsed.document_type,)
    return parsed


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def _mb(value: int | None) -> float | None:
    return round(value / 1024 / 1024, 3) if value is not None else None


def _git(*command: str) -> str:
    return subprocess.run(["git", *command], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()


def _environment(cli: argparse.Namespace, manifest: dict[str, Any], configuration: dict[str, str]) -> dict[str, Any]:
    versions = {}
    for name in ("fastapi", "uvicorn", "requests", "numpy", "opencv-python", "paddleocr", "paddlepaddle", "onnxruntime"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    cpu_model = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                cpu_model = line.split(":", 1)[1].strip()
                break
    except (FileNotFoundError, IndexError):
        pass
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cpu_only": True,
        "git": {"sha": _git("rev-parse", "HEAD"), "dirty": bool(_git("status", "--porcelain")), "status": _git("status", "--short")},
        "dataset": {"root": str(cli.dataset_root.resolve()), "manifest_sha256": hashlib.sha256(_json(manifest).encode()).hexdigest()},
        "configuration": configuration,
        "model_environment": MODEL_ENV,
        "runtime": {"python": sys.version, "implementation": platform.python_implementation(), "platform": platform.platform(), "machine": platform.machine(), "cpu_model": cpu_model, "cpu_count": os.cpu_count()},
        "packages": versions,
        "benchmark_cli": {key: str(value) for key, value in vars(cli).items() if key != "kinds"},
    }


def _server_env(model_dir: Path, trace_dir: Path, artifact_dir: Path | None = None, batch_overrides: dict[str, str] | None = None) -> dict[str, str]:
    environment = {
        "RUNTIME_TARGET": "cpu",
        "OCR_DEVICE": "cpu",
        "MODEL_DIR": str(model_dir.resolve()),
        "PRELOAD": "true",
        "LOGGING": "false",
        "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
        "OMP_NUM_THREADS": "1",
        "CPU_THREADS": "4",
        "LOCALIZATION_BATCH_SIZE": "4",
        "TEXT_DETECTION_BATCH_SIZE": "1",
        "TEXT_RECOGNITION_BATCH_SIZE": "2",
        "MRZ_RECOGNITION_BATCH_SIZE": "2",
        "TEXT_RECOGNITION_PROCESSES": "1",
        "TEXT_RECOGNITION_PACKING": "fixed-width",
        **MODEL_ENV,
        "VOIGHT_BENCHMARK_TRACE_DIR": str(trace_dir),
    }
    if batch_overrides:
        environment.update(batch_overrides)
    if artifact_dir is not None:
        environment["VOIGHT_BENCHMARK_ARTIFACT_DIR"] = str(artifact_dir)
    return environment


def _mime(path: Path) -> str:
    return "image/png" if path.suffix.lower() == ".png" else "image/jpeg"


def _files(document: Document) -> dict[str, tuple[str, bytes, str]]:
    return {role: (path.name, path.read_bytes(), _mime(path)) for role, path in document.paths}


def _values(document: Document) -> dict[str, Any]:
    return {
        name: field["value"]
        for name, field in annotation_truth(document).get("fields", {}).items()
        if isinstance(field, dict) and field.get("state") == "value" and field.get("value") not in (None, "")
    }


def _trace(trace_dir: Path, operation: str, before: set[Path]) -> dict[str, Any]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        paths = set(trace_dir.glob(f"{operation}-*.json")) - before
        if paths:
            return json.loads(next(iter(paths)).read_text(encoding="utf-8"))
        time.sleep(0.01)
    raise RuntimeError(f"missing {operation} benchmark trace in {trace_dir}")


def _stage_elapsed(stage: Any) -> float:
    return float(stage.get("elapsed_wall_seconds", stage.get("wall_seconds", 0.0))) if isinstance(stage, dict) else 0.0


def _batch_detail(stage: Any) -> dict[str, Any]:
    if not isinstance(stage, dict):
        return {}
    calls = stage.get("calls", [])
    tensor = [size for call in calls for size in call.get("tensor_batch_sizes", [call.get("tensor_batch_size")]) if size is not None]
    return {"configured_batch_size": stage.get("configured_batch_size"), "tensor_batch_sizes": tensor, "submitted_batch_sizes": [call.get("submitted_batch_size") for call in calls], "model_call_count": stage.get("model_call_count", len(calls)), "calls": calls}


def _ocr_detail(trace: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    diagnostics = trace.get("diagnostics", {})
    lines = payload.get("lines", payload.get("front", []) + payload.get("back", []))
    confidences = [float(line["confidence"]) for line in lines]
    line_filter = diagnostics.get("line_filter", {})
    line_crop = float(diagnostics.get("line_crop_seconds", 0.0))
    result_unpack = float(diagnostics.get("result_unpack_seconds", 0.0))
    return {
        "stages": {
            "input_decode": float(trace.get("image_decode_seconds", 0.0)),
            "model_inference": float(trace.get("model_inference_seconds", 0.0)),
            "text_detection": _stage_elapsed(diagnostics.get("text_detection")),
            "text_recognition": _stage_elapsed(diagnostics.get("text_recognition")),
            "ocr_line_crop": line_crop,
            "ocr_result_unpack": result_unpack,
            "ocr_post_processing_ordering": line_crop + result_unpack,
            "response_construction": float(trace.get("response_assembly_seconds", 0.0)),
        },
        "detected_line_count": int(line_filter.get("detected_line_count", len(lines))),
        "recognition_candidate_count": int(line_filter.get("recognition_candidate_count", len(lines))),
        "filtered_before_recognition_count": int(line_filter.get("filtered_before_recognition_count", 0)),
        "ocr_confidence_mean": statistics.mean(confidences) if confidences else None,
        "ocr_confidence_min": min(confidences) if confidences else None,
        "ocr_confidence_max": max(confidences) if confidences else None,
        "actual_tensor_batches": {name: _batch_detail(diagnostics.get(name)) for name in ("text_detection", "text_recognition")},
        "diagnostics": diagnostics,
    }


def _check_detail(trace: dict[str, Any], payload: dict[str, Any]) -> dict[str, Any]:
    value = next(iter(payload.get("fields", {}).values()), {})
    return {
        "stages": {
            "verification_normalization": float(trace.get("normalization_and_line_conversion_seconds") or 0.0) + float(trace.get("normalization_seconds") or 0.0),
            "candidate_construction": float(trace.get("candidate_generation_seconds") or 0.0),
            "fuzzy_similarity_matching": float(trace.get("similarity_scoring_seconds") or 0.0),
            "global_assignment": float(trace.get("assignment_seconds") or 0.0),
            "status_classification": float(trace.get("status_classification_seconds") or 0.0),
            "response_construction": float(trace.get("response_assembly_seconds") or 0.0),
        },
        "line_count": int(trace.get("line_count", 0)),
        "candidate_count": int(trace.get("candidate_count", 0)),
        "score_comparison_count": int(trace.get("score_comparison_count", 0)),
        "assignment_states": int(trace.get("assignment_states", 0)),
        "assignment_transitions": int(trace.get("assignment_transitions", 0)),
        "result": value,
        "trace": trace,
    }


def _request(server_url: str, path: str, files: dict[str, Any], trace_dir: Path, operation: str, json_body: dict[str, Any] | None, timeout: float) -> tuple[dict[str, Any], dict[str, Any] | None]:
    before = set(trace_dir.glob(f"{operation}-*.json"))
    started = time.perf_counter()
    try:
        response = requests.post(f"{server_url}{path}", files=None if json_body is not None else files, json=json_body, timeout=timeout)
        elapsed = time.perf_counter() - started
    except requests.RequestException as error:
        return {"status": "request_failed", "error": f"{type(error).__name__}: {error}", "client_latency_seconds": time.perf_counter() - started}, None
    if not response.ok:
        return {"status": "request_failed", "http_status": response.status_code, "error": response.text[:1000], "client_latency_seconds": elapsed}, None
    try:
        payload = response.json()
        trace = _trace(trace_dir, operation, before)
    except (ValueError, RuntimeError) as error:
        return {"status": "invalid_response", "http_status": response.status_code, "error": str(error), "client_latency_seconds": elapsed}, None
    return {"status": "ok", "http_status": response.status_code, "client_latency_seconds": elapsed, "server_latency_seconds": float(trace.get("total_server_seconds", 0.0)), "response": payload, "trace": trace}, payload


def _safe_name(value: str) -> str:
    return "".join(character if character.isalnum() or character in "_.-" else "-" for character in value).strip(".-") or "document"


def _move_visual_artifact(root: Path, target: Path, before: set[Path]) -> Path | None:
    if not root.exists():
        return None
    candidates = set(root.glob("ocr-*")) - before
    if not candidates:
        return None
    source = max(candidates, key=lambda path: path.stat().st_mtime_ns)
    target.parent.mkdir(parents=True, exist_ok=True)
    source.rename(target)
    sample_dirs = sorted(path for path in target.iterdir() if path.is_dir() and "_" in path.name)
    if len(sample_dirs) == 1:
        sample_dir = sample_dirs[0]
        for child in sample_dir.iterdir():
            child.rename(target / child.name)
        sample_dir.rmdir()
    elif len(sample_dirs) > 1:
        for sample_dir in sample_dirs:
            sample_dir.rename(target / sample_dir.name.split("_", 1)[1])
    manifest = target / "manifest.json"
    if manifest.exists():
        value = json.loads(manifest.read_text(encoding="utf-8"))
        for sample in value.get("samples", []):
            directory = sample.get("directory", "")
            sample["directory"] = "." if len(sample_dirs) == 1 else directory.split("_", 1)[-1]
        manifest.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    return target


def _write_ocr_human_artifacts(document_dir: Path, payload: dict[str, Any]) -> None:
    (document_dir / "ocr_response.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = payload.get("lines", [])
    if not lines:
        lines = [line for side in ("front", "back") for line in payload.get(side, [])]
    output = ["# OCR result", "", "Open `detection.png` first, then `recognition_contact_sheet.png`.", "", "| Line | Recognized text | Confidence |", "|---:|---|---:|"]
    for index, line in enumerate(lines, 1):
        text_value = str(line.get("text", "")).replace("|", "\\|")
        output.append(f"| {index} | {text_value} | {line.get('confidence')} |")
    (document_dir / "ocr_summary.md").write_text("\n".join(output) + "\n", encoding="utf-8")


def _write_fuzzy_artifacts(document_dir: Path, document: Document, checks: list[dict[str, Any]]) -> None:
    document_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for request in checks:
        result = request.get("result", {})
        trace = request.get("trace", {})
        evidence = next((item for item in trace.get("candidate_evidence", []) if item.get("field") == request.get("field")), {"candidates": []})
        records.append({"field": request.get("field"), "expected": request.get("expected"), "result": result, "candidate_evidence": evidence, "stages": request.get("stages", {}), "trace": trace})
    (document_dir / "fuzzy_matching.json").write_text(json.dumps({"document_type": document.document_type, "document_id": document.document_id, "fields": records}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    lines = ["# Fuzzy matching", "", "Read each field from left to right: expected truth → selected OCR value → final status. The candidate table shows every score considered by global assignment.", "", "Status: `match` = normalized exact; `likely_match` = accepted fuzzy match; `mismatch` = a candidate was selected but failed the threshold; `not_found` = no eligible candidate."]
    for record in records:
        result = record["result"]
        lines += ["", f"## `{record['field']}`", "", f"- Expected: `{record['expected']}`", f"- Selected OCR: `{result.get('detected')}`", f"- Score: `{result.get('score')}`", f"- Status: **{result.get('status', 'request_failed')}**", f"- Source: `{result.get('source')}`", "", "| Rank | Candidate | Score | Eligible | Selected |", "|---:|---|---:|:---:|:---:|"]
        candidates = record["candidate_evidence"].get("candidates", [])
        for rank, candidate in enumerate(candidates, 1):
            selected = candidate.get("text") == result.get("detected") and result.get("detected") is not None
            text_value = str(candidate.get("text", "")).replace("|", "\\|")
            lines.append(f"| {rank} | {text_value} | {candidate.get('score')} | {candidate.get('eligible')} | {selected} |")
    (document_dir / "fuzzy_matching.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    overview = [f"# {_safe_name(document.document_type)} / {document.document_id}", "", "## Inspection order", "", "1. `detection.png`: are the text boxes on the right words?", "2. `recognition_contact_sheet.png`: does each crop read correctly?", "3. `ocr_summary.md`: what text and confidence reached the API?", "4. `fuzzy_matching.md`: why was each candidate accepted or rejected?", "", f"Fields checked: {len(records)}."]
    (document_dir / "README.md").write_text("\n".join(overview) + "\n", encoding="utf-8")


def _ocr(server_url: str, document: Document, repeat: int, server: Server, trace_dir: Path, timeout: float, artifact_root: Path | None = None, artifact_target: Path | None = None) -> tuple[dict[str, Any], dict[str, Any] | None]:
    _, ocr_path, _ = ROUTES[document.document_type]
    artifact_before = set(artifact_root.glob("ocr-*")) if artifact_root is not None else set()
    request, payload = _request(server_url, ocr_path, _files(document), trace_dir, "ocr", None, timeout)
    request.update({"phase": "ocr", "document_type": document.document_type, "document_id": document.document_id, "repeat": repeat})
    if payload is not None:
        request.update(_ocr_detail(request["trace"], payload))
    if artifact_root is not None and artifact_target is not None:
        visual_dir = _move_visual_artifact(artifact_root, artifact_target, artifact_before)
        if visual_dir is not None and payload is not None:
            _write_ocr_human_artifacts(visual_dir, payload)
            request["visual_artifact_dir"] = str(visual_dir)
    server._sample()
    return request, payload


def _check(server_url: str, document: Document, ocr_payload: dict[str, Any], field: str, expected: Any, repeat: int, trace_dir: Path, timeout: float) -> tuple[dict[str, Any], dict[str, Any] | None]:
    _, _, check_path = ROUTES[document.document_type]
    request, payload = _request(server_url, check_path, {}, trace_dir, "check", {"ocr": ocr_payload, "fields": {field: expected}}, timeout)
    request.update({"phase": "check", "document_type": document.document_type, "document_id": document.document_id, "field": field, "repeat": repeat, "expected": str(expected)})
    if payload is not None:
        request.update(_check_detail(request["trace"], payload))
    return request, payload


class RssMonitor:
    def __init__(self, server: Server, interval: float):
        self.server, self.interval = server, interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="verification-rss", daemon=True)
        self.started = False

    def start(self) -> None:
        self.started = True
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            self.server._sample()
            self.stop_event.wait(self.interval)

    def stop(self) -> None:
        if not self.started:
            return
        self.stop_event.set()
        self.thread.join(timeout=max(1.0, self.interval * 5))


def _field_row(request: dict[str, Any], document: Document) -> dict[str, Any]:
    result = request.get("result", {})
    detected = result.get("detected")
    status = result.get("status")
    return {
        "repeat": request["repeat"], "document_type": document.document_type, "document_id": document.document_id, "field": request["field"], "expected": request["expected"], "matched_ocr_value": detected, "score": result.get("score"), "status": status or "request_failed", "source": result.get("source"), "score_source": result.get("score_source"), "raw_exact_match": detected == request["expected"], "normalized_exact_match": status == "match", "request_status": request["status"], "client_latency_seconds": request.get("client_latency_seconds"), "server_latency_seconds": request.get("server_latency_seconds"), "stages": request.get("stages", {}), "candidate_count": request.get("candidate_count"), "score_comparison_count": request.get("score_comparison_count"), "assignment_states": request.get("assignment_states"), "assignment_transitions": request.get("assignment_transitions"), "ocr_response": request.get("ocr_payload"), "verification_response": request.get("response"),
    }


def _accuracy(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successful = [row for row in rows if row["request_status"] == "ok" and row["status"] in {"match", "likely_match", "mismatch", "not_found"}]
    scores = [float(row["score"]) for row in successful if row.get("score") is not None]
    counts = {name: sum(row["status"] == name for row in successful) for name in ("match", "likely_match", "mismatch", "not_found")}
    total = len(rows)
    evaluated = len(successful)
    accepted = counts["match"] + counts["likely_match"]
    return {"total_truth_fields": total, "evaluated_fields": evaluated, "failed_field_requests": total - evaluated, "raw_exact_matches": sum(bool(row["raw_exact_match"]) for row in successful), "normalized_exact_matches": counts["match"], "accepted_fuzzy_matches": counts["likely_match"], "mismatches": counts["mismatch"], "not_found": counts["not_found"], "field_verification_accuracy": accepted / evaluated if evaluated else None, "field_verification_coverage": evaluated / total if total else None, "normalized_exact_accuracy": counts["match"] / evaluated if evaluated else None, "mean_matching_score": statistics.mean(scores) if scores else None, "median_matching_score": statistics.median(scores) if scores else None, **counts}


def _summary_rows(request_rows: list[dict[str, Any]], field_rows: list[dict[str, Any]], documents: list[Document], outcomes: list[dict[str, Any]], run_rows: list[dict[str, Any]], repeats: int) -> list[dict[str, Any]]:
    rows = []
    groups = [(kind, [document for document in documents if document.document_type == kind]) for kind in KINDS]
    groups.append(("overall", documents))
    for kind, group_documents in groups:
        request_values = [row for row in request_rows if kind == "overall" or row.get("document_type") == kind]
        field_values = [row for row in field_rows if kind == "overall" or row["document_type"] == kind]
        outcome_values = [row for row in outcomes if kind == "overall" or row["document_type"] == kind]
        ocr = [row for row in request_values if row.get("phase") == "ocr" and row.get("status") == "ok"]
        checks = [row for row in request_values if row.get("phase") == "check" and row.get("status") == "ok"]
        accuracy = _accuracy(field_values)
        rows.append({
            "document_type": kind, "documents": len(group_documents), "physical_images": sum(document.physical_count for document in group_documents), "repeats": repeats,
            "ocr_requests": sum(row.get("phase") == "ocr" for row in request_values), "ocr_succeeded": len(ocr), "ocr_failed": sum(row.get("phase") == "ocr" and row.get("status") != "ok" for row in request_values),
            "check_requests": sum(row.get("phase") == "check" for row in request_values), "check_succeeded": len(checks), "check_failed": sum(row.get("phase") == "check" and row.get("status") != "ok" for row in request_values),
            "median_ocr_client_seconds": statistics.median(row["client_latency_seconds"] for row in ocr) if ocr else None, "median_ocr_server_seconds": statistics.median(row["server_latency_seconds"] for row in ocr) if ocr else None,
            "median_check_client_seconds": statistics.median(row["client_latency_seconds"] for row in checks) if checks else None, "median_check_server_seconds": statistics.median(row["server_latency_seconds"] for row in checks) if checks else None,
            "median_run_documents_per_second": statistics.median((len(group_documents) / row["client_latency_seconds"]) for row in run_rows if row.get("status") == "ok") if group_documents and run_rows else None,
            "ocr_stage_medians": {stage: statistics.median(row["stages"].get(stage, 0.0) for row in ocr) if ocr else None for stage in ("input_decode", "model_inference", "text_detection", "text_recognition", "ocr_post_processing_ordering", "response_construction")},
            "check_stage_medians": {stage: statistics.median(row["stages"].get(stage, 0.0) for row in checks) if checks else None for stage in ("verification_normalization", "candidate_construction", "fuzzy_similarity_matching", "global_assignment", "status_classification", "response_construction")},
            "document_fully_correct": sum(bool(row["fully_correct"]) for row in outcome_values), "document_fully_correct_rate": sum(bool(row["fully_correct"]) for row in outcome_values) / len(outcome_values) if outcome_values else None, "document_evaluated": len(outcome_values), "documents_with_accepted_fuzzy": sum(bool(row["accepted_fuzzy"]) for row in outcome_values), **accuracy,
        })
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json(value) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def _report(path: Path, environment: dict[str, Any], manifest: dict[str, Any], summary: list[dict[str, Any]], lifecycle: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    configuration = environment["configuration"]
    lines = [
        "# Verification benchmark",
        "",
        f"- CPU-only: `{environment['cpu_only']}`",
        f"- Git: `{environment['git']['sha']}` (dirty: `{environment['git']['dirty']}`)",
        f"- Dataset: `{_json(manifest['counts'])}` logical documents, `{manifest['physical_images']}` physical images",
        f"- Models: detector `{configuration['TEXT_DETECTOR_MODEL']}`, recognizer `{configuration['TEXT_RECOGNIZER_MODEL']}`, MRZ backend `{configuration['MRZ_RECOGNIZER_BACKEND']}`",
        f"- Threads/batches: CPU threads `{configuration['CPU_THREADS']}`, detection `{configuration['TEXT_DETECTION_BATCH_SIZE']}`, recognition `{configuration['TEXT_RECOGNITION_BATCH_SIZE']}`",
        "- Procedure: one warm-up plus three measured repeats; warm-up excluded; fresh API process for every repeat",
        "",
        "## Results",
        "",
        "| Type | OCR median client / server (s) | Check median client / server (s) | Throughput (docs/s) | Exact / fuzzy / mismatch / not-found | Accepted accuracy | Coverage | Fully-correct |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary:
        lines.append(f"| {row['document_type']} | {row['median_ocr_client_seconds']:.4f} / {row['median_ocr_server_seconds']:.4f} | {row['median_check_client_seconds']:.4f} / {row['median_check_server_seconds']:.4f} | {row['median_run_documents_per_second']:.4f} | {row['normalized_exact_matches']} / {row['accepted_fuzzy_matches']} / {row['mismatches']} / {row['not_found']} | {row['field_verification_accuracy']:.2%} | {row['field_verification_coverage']:.2%} | {row['document_fully_correct']} / {row['document_evaluated']} |" if row["median_ocr_client_seconds"] is not None else f"| {row['document_type']} | n/a | n/a | n/a | 0 / 0 / 0 / 0 | n/a | n/a | 0 / 0 |")
    overall = next(row for row in summary if row["document_type"] == "overall")
    bottleneck = max(((stage, value) for stage, value in overall["ocr_stage_medians"].items() if value is not None), key=lambda item: item[1], default=("none", 0.0))
    lines += ["", "## Stage medians (seconds)", "", "| Type | OCR input | Detection | Recognition | OCR ordering | Verification fuzzy | Assignment |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in summary:
        ocr_stages = row["ocr_stage_medians"]
        check_stages = row["check_stage_medians"]
        lines.append(f"| {row['document_type']} | {ocr_stages['input_decode']:.4f} | {ocr_stages['text_detection']:.4f} | {ocr_stages['text_recognition']:.4f} | {ocr_stages['ocr_post_processing_ordering']:.4f} | {check_stages['fuzzy_similarity_matching']:.4f} | {check_stages['global_assignment']:.4f} |" if ocr_stages["input_decode"] is not None else f"| {row['document_type']} | n/a | n/a | n/a | n/a | n/a | n/a |")
    lines += ["", "## RAM and lifecycle", "", "| Repeat | Baseline RSS (MB) | Peak RSS (MB) | After shutdown | Cleanup verified | RSS released |", "|---:|---:|---:|---:|:---:|:---:|"]
    for record in lifecycle:
        lines.append(f"| {record['repeat']} | {record.get('server_rss_baseline_mb')} | {record.get('peak_process_memory_mb')} | {record.get('server_rss_after_shutdown_mb')} | {record.get('cleanup_verified')} | {record.get('rss_released')} |")
    lines += ["", "## Failures and bottleneck", "", f"- Request/run failures: `{len(failures)}`", f"- Largest OCR median stage: `{bottleneck[0]}` at `{bottleneck[1]:.4f}s`", f"- Overall accepted field accuracy: `{overall['field_verification_accuracy']:.2%}`; normalized exact: `{overall['normalized_exact_accuracy']:.2%}`; coverage: `{overall['field_verification_coverage']:.2%}`", f"- Mean / median matching score: `{overall['mean_matching_score']:.4f}` / `{overall['median_matching_score']:.4f}`", "", "Open `README.md` and `analysis.md` for the human triage path. Raw responses, traces, tensor batches, and full lifecycle records remain in the JSON/CSV files."]
    for failure in failures[:20]:
        lines.append(f"- `{failure}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_human_guides(output: Path, manifest: dict[str, Any], summary: list[dict[str, Any]], field_rows: list[dict[str, Any]]) -> None:
    overall = next(row for row in summary if row["document_type"] == "overall")
    worst = sorted(
        (row for row in field_rows if row.get("status") in {"mismatch", "not_found"}),
        key=lambda row: (row.get("score") is not None, row.get("score") or 0.0),
    )[:30]
    guide = [
        "# How to inspect this benchmark",
        "",
        "Start with `report.md` for the decision summary. For visual diagnosis, open one folder under `01.repeat-1/batch/`:",
        "",
        "```text",
        "01.repeat-1/batch/001/passport-p_1/",
        "├── source.png                    original upload",
        "├── detection.png                 boxes and D001, D002… labels",
        "├── recognition_contact_sheet.png readable crop overview",
        "├── recognition/                  individual raw/processed crops",
        "├── ocr_summary.md                OCR text and confidence",
        "└── fuzzy_matching.md             every candidate score and final status",
        "```",
        "",
        "For an ID card, the same files are inside `front/` and `back/` under the document folder.",
        "",
        "`02.repeat-2/` and `03.repeat-3/` have the same structure. Warm-up artifacts are under each `NN.repeat-*/warmup/` and are excluded from statistics.",
        "",
        "## What needs attention",
        "",
        f"Overall normalized-exact accuracy is `{overall['normalized_exact_accuracy']}`; accepted verification accuracy is `{overall['field_verification_accuracy']}`; document fully-correct rate is `{overall['document_fully_correct_rate']}`.",
        "",
        "The rows below are the first fields to inspect. Find the matching document folder, then follow detection → recognition → fuzzy matching.",
        "",
        "| Type / document / repeat | Field | Expected | OCR value | Score | Status |",
        "|---|---|---|---|---:|---|",
    ]
    for row in worst:
        guide.append(f"| {row['document_type']} / {row['document_id']} / repeat-{row['repeat']} | {row['field']} | {row['expected']} | {row['matched_ocr_value']} | {row['score']} | {row['status']} |")
    if not worst:
        guide += ["| — | — | — | — | — | no mismatch/not-found fields |"]
    guide += ["", "## Decision guide", "", "- `not_found`: inspect `detection.png` first; a missing or misplaced box usually explains it.", "- `mismatch`: inspect the corresponding recognition crop and then `fuzzy_matching.md` to see whether the right candidate lost global assignment.", "- High OCR confidence with a wrong value points to recognition or profile/field association; low confidence points to image quality, detection, or preprocessing.", "- Compare the same document across repeats only for stability; warm-up is not measured.", ""]
    (output / "README.md").write_text("\n".join(guide), encoding="utf-8")

    analysis = ["# Analysis notes", "", "This file is the compact human triage view. Raw evidence remains in `raw_runs.jsonl`, `field_results.csv`, per-request traces, and each document folder.", "", "## Counts", "", f"- Documents: `{len(manifest['documents'])}`; physical images: `{manifest['physical_images']}`", f"- Truth fields measured: `{overall['total_truth_fields']}`", f"- Exact matches: `{overall['normalized_exact_matches']}`; accepted fuzzy: `{overall['accepted_fuzzy_matches']}`; mismatches: `{overall['mismatches']}`; not-found: `{overall['not_found']}`", f"- Request failures: `{overall['failed_field_requests']}`", "", "## First fixes to investigate", ""]
    for row in worst[:10]:
        analysis.append(f"- `{row['document_type']}/{row['document_id']}` field `{row['field']}`: `{row['status']}`, expected `{row['expected']}`, OCR `{row['matched_ocr_value']}`, score `{row['score']}`. Inspect `{row['repeat']:02d}.repeat-{row['repeat']}/batch/` for the images and `fuzzy_matching.md`.")
    if not worst:
        analysis.append("- No mismatch or not-found fields were measured.")
    (output / "analysis.md").write_text("\n".join(analysis) + "\n", encoding="utf-8")


def main() -> int:
    cli = args()
    documents, manifest = validate_and_manifest(cli.dataset_root)
    selected = [document for kind in cli.kinds for document in documents if document.document_type == kind][: cli.limit] if cli.limit is not None else [document for document in documents if document.document_type in cli.kinds]
    if not selected:
        raise RuntimeError("no valid documents selected")
    selected_ids = {document.document_id for document in selected}
    selected_manifest = {**manifest, "documents": [entry for entry in manifest["documents"] if entry["document_id"] in selected_ids], "counts": {kind: sum(document.document_type == kind for document in selected) for kind in DOC_TYPES}, "physical_images": sum(document.physical_count for document in selected)}
    output = cli.output_dir or cli.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    (output / "server_logs").mkdir()
    configuration = _server_env(cli.model_dir, output / "trace", output / "NN.repeat-*/_visual")
    environment = _environment(cli, selected_manifest, configuration)
    (output / "environment.json").write_text(json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "dataset_manifest.json").write_text(json.dumps(selected_manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    request_rows: list[dict[str, Any]] = []
    field_rows: list[dict[str, Any]] = []
    outcomes: list[dict[str, Any]] = []
    lifecycle_rows: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for repeat in range(1, cli.repeats + 1):
        repeat_dir = output / f"{repeat:02d}.repeat-{repeat}"
        trace_dir = repeat_dir / "trace"
        trace_dir.mkdir(parents=True)
        visual_root = repeat_dir / "_visual"
        server = VerificationServer(cli, repeat_dir, _server_env(cli.model_dir, trace_dir, visual_root))
        monitor = RssMonitor(server, cli.rss_interval)
        lifecycle: dict[str, Any] = {"repeat": repeat, "measured": False, "memory_before_mb": _mb(_available_memory()), "server_rss_baseline_mb": None}
        measured_started = None
        try:
            startup_started = time.perf_counter()
            ready = server.start()
            lifecycle.update({"startup_client_seconds": time.perf_counter() - startup_started, "server_pid": server.process.pid if server.process else None, "server_rss_baseline_mb": _mb(_rss(server.process.pid)) if server.process else None, "effective_configuration": ready})
            (repeat_dir / "ready.json").write_text(json.dumps(ready, indent=2, ensure_ascii=False), encoding="utf-8")
            monitor.start()
            warmup_rows = []
            warmup_started = time.perf_counter()
            for document_index, document in enumerate(selected, 1):
                warmup_target = repeat_dir / "warmup" / "batch" / f"{document_index:03d}" / f"{document.document_type}-{_safe_name(document.document_id)}"
                warmup_ocr, warmup_payload = _ocr(f"http://127.0.0.1:{cli.port}", document, 0, server, trace_dir, cli.timeout, visual_root, warmup_target)
                warmup_rows.append(warmup_ocr)
                if warmup_payload is not None:
                    values = _values(document)
                    field = next(iter(values), None)
                    if field is not None:
                        warmup_check, _ = _check(f"http://127.0.0.1:{cli.port}", document, warmup_payload, field, values[field], 0, trace_dir, cli.timeout)
                        warmup_rows.append(warmup_check)
            (repeat_dir / "warmup.json").write_text(json.dumps({"policy": "one OCR and first-field check per selected document", "client_seconds": time.perf_counter() - warmup_started, "requests": warmup_rows}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
            measured_started = time.perf_counter()
            lifecycle["measured"] = True
            for document_index, document in enumerate(selected, 1):
                document_dir = repeat_dir / "batch" / f"{document_index:03d}" / f"{document.document_type}-{_safe_name(document.document_id)}"
                ocr_request, ocr_payload = _ocr(f"http://127.0.0.1:{cli.port}", document, repeat, server, trace_dir, cli.timeout, visual_root, document_dir)
                request_rows.append(ocr_request)
                if ocr_payload is None:
                    failures.append({"phase": "ocr", "document_id": document.document_id, "repeat": repeat, "error": ocr_request.get("error")})
                    continue
                statuses = []
                document_checks = []
                for field, expected in _values(document).items():
                    check_request, check_payload = _check(f"http://127.0.0.1:{cli.port}", document, ocr_payload, field, expected, repeat, trace_dir, cli.timeout)
                    check_request["ocr_payload"] = ocr_payload
                    request_rows.append(check_request)
                    document_checks.append(check_request)
                    if check_payload is None:
                        failures.append({"phase": "check", "document_id": document.document_id, "field": field, "repeat": repeat, "error": check_request.get("error")})
                    row = _field_row(check_request, document)
                    field_rows.append(row)
                    statuses.append(row["status"] if row["request_status"] == "ok" else "request_failed")
                _write_fuzzy_artifacts(document_dir, document, document_checks)
                outcomes.append({"repeat": repeat, "document_type": document.document_type, "document_id": document.document_id, "fully_correct": bool(statuses) and all(status == "match" for status in statuses), "accepted_fuzzy": bool(statuses) and all(status in {"match", "likely_match"} for status in statuses), "field_statuses": statuses})
            elapsed = time.perf_counter() - measured_started
            run_rows.append({"phase": "run", "repeat": repeat, "status": "ok", "client_latency_seconds": elapsed, "documents": len(selected), "throughput_documents_per_second": len(selected) / elapsed})
            (repeat_dir / "measured.json").write_text(json.dumps({"requests": [row for row in request_rows if row.get("repeat") == repeat], "document_outcomes": [row for row in outcomes if row.get("repeat") == repeat]}, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        except Exception as error:
            failures.append({"phase": "run", "repeat": repeat, "error": f"{type(error).__name__}: {error}"})
            lifecycle["error"] = failures[-1]["error"]
        finally:
            monitor.stop()
            cleanup = server.stop()
            lifecycle.update(cleanup)
            lifecycle["memory_after_shutdown_mb"] = _mb(_available_memory())
            lifecycle["server_rss_after_shutdown_mb"] = _mb(_rss(lifecycle.get("server_pid"))) if lifecycle.get("server_pid") else None
            lifecycle["process_group_empty"] = not _pids_in_group(lifecycle["server_pgid"]) if lifecycle.get("server_pgid") else True
            lifecycle["rss_released"] = lifecycle["server_rss_after_shutdown_mb"] is None and lifecycle["process_group_empty"]
            lifecycle_rows.append(lifecycle)
            (repeat_dir / "lifecycle.json").write_text(json.dumps(lifecycle, indent=2, ensure_ascii=False), encoding="utf-8")
        if not lifecycle.get("cleanup_verified", False):
            failures.append({"phase": "cleanup", "repeat": repeat, "error": "API or child process remained after shutdown"})
            break
    (output / "raw_runs.jsonl").write_text("\n".join(_json(row) for row in request_rows + run_rows) + "\n", encoding="utf-8")
    _write_csv(output / "raw_runs.csv", request_rows + run_rows)
    _write_csv(output / "field_results.csv", field_rows)
    summary = _summary_rows(request_rows, field_rows, selected, outcomes, run_rows, cli.repeats)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    _write_csv(output / "summary.csv", summary)
    (output / "failures.json").write_text(json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8")
    _report(output / "report.md", environment, selected_manifest, summary, lifecycle_rows, failures)
    _write_human_guides(output, selected_manifest, summary, field_rows)
    print(f"completed verification benchmark: {output}")
    return 2 if failures or not all(row.get("cleanup_verified", False) for row in lifecycle_rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
