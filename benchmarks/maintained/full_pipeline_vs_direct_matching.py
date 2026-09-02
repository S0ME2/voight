"""CPU-only apples-to-apples benchmark of the two real extraction routes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import signal
import socket
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from benchmarks.maintained.pipeline_breakdown import Document, annotation_truth, validate_and_manifest

FULL = "FULL_LATIN_PIPELINE"
DIRECT = "DIRECT_MATCHING_PIPELINE"
KINDS = ("passport", "id_card", "driving_license")
ROUTES = {
    "passport": ("passport", "passport"),
    "id_card": ("id-card", "id-card"),
    "driving_license": ("driving-license", "driving-licence"),
}
DETECTOR = "PP-OCRv6_medium_det"
RECOGNIZER = "latin_PP-OCRv5_mobile_rec"
MODEL_ENV = {
    "TEXT_DETECTOR_BACKEND": "paddle",
    "TEXT_DETECTOR_MODEL": DETECTOR,
    "TEXT_RECOGNIZER_BACKEND": "paddle",
    "TEXT_RECOGNIZER_MODEL": RECOGNIZER,
    "DOCALIGNER_MODEL": "fastvit_sa24",
    "DOCALIGNER_MODEL_TYPE": "heatmap",
    "MRZSCANNER_DETECTION_CFG": "20250222",
    "MRZ_RECOGNIZER_BACKEND": "generic-paddle",
    "MRZ_RECOGNIZER_MODEL": "20250221",
}
BASE_ENV = {
    "RUNTIME_TARGET": "cpu",
    "OCR_DEVICE": "cpu",
    "PRELOAD": "false",
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
}
STRICT_FIELDS = {
    "passport_number", "card_number", "pinfl", "personal_id", "license_number", "serial_number",
    "date_of_birth", "date_of_issue", "date_of_expiry", "birth_date", "issue_date", "expiry_date",
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, separators=(",", ":"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json(value) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()})


def _port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def _pids_in_group(pgid: int) -> list[int]:
    result = []
    for path in Path("/proc").glob("[0-9]*"):
        try:
            stat = path.joinpath("stat").read_text().split(") ", 1)[1].split()
            if int(stat[2]) == pgid:
                result.append(int(path.name))
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            pass
    return result


def _rss(pid: int | None) -> int | None:
    if not pid:
        return None
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return None


def _mime(path: Path) -> str:
    return "image/png" if path.suffix.lower() == ".png" else "image/jpeg"


def _files(document: Document) -> dict[str, tuple[str, bytes, str]]:
    return {role: (path.name, path.read_bytes(), _mime(path)) for role, path in document.paths}


def _truth(document: Document) -> dict[str, str]:
    return {
        name: str(entry.get("value", ""))
        for name, entry in annotation_truth(document).get("fields", {}).items()
        if isinstance(entry, dict) and entry.get("state") == "value" and entry.get("value") not in (None, "")
    }


def _strip_label(value: str) -> str:
    import re
    return re.sub(r"^\s*(?:(?:\d+[A-Z]|[A-Z]\d*)[.)]|\d+[.)])\s*", "", value, count=1, flags=re.I)


def _neutral_normalize(value: Any, field: str) -> str:
    """Frozen evaluator: normalized equality only; never invokes the matcher."""
    import re
    from datetime import date

    text = _strip_label(str(value or "")).upper()
    kind = "date" if "date" in field or field in {"birth_date", "issue_date", "expiry_date"} else "identifier" if field in STRICT_FIELDS or any(word in field for word in ("number", "pinfl", "personal_id", "serial")) else "text"
    if kind == "date":
        parts = re.split(r"[./, -]+", text.strip())
        if len(parts) == 1 and parts[0].isdigit() and len(parts[0]) in {6, 8}:
            parts = [parts[0][:2], parts[0][2:4], parts[0][4:]]
        if len(parts) == 3 and all(part.isdigit() for part in parts):
            try:
                numbers = tuple(map(int, parts))
                year = numbers[0] if len(parts[0]) == 4 else numbers[2] + (2000 if numbers[2] < 100 else 0)
                return date(year, numbers[1], numbers[2] if len(parts[0]) == 4 else numbers[0]).isoformat()
            except ValueError:
                pass
        return re.sub(r"\D", "", text)
    if kind == "identifier":
        return "".join(char for char in text if char.isalnum())
    return " ".join(re.sub(r"[^\w ]", " ", text, flags=re.UNICODE).split())


def _field_status(expected: str, detected: Any, field: str) -> str:
    if detected in (None, ""):
        return "not_found"
    return "correct" if _neutral_normalize(expected, field) == _neutral_normalize(detected, field) else "incorrect"


def _trace(trace_dir: Path, operation: str, before: set[Path]) -> tuple[dict[str, Any], Path]:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        paths = set(trace_dir.glob(f"{operation}-*.json")) - before
        if paths:
            path = max(paths, key=lambda item: item.stat().st_mtime_ns)
            return json.loads(path.read_text(encoding="utf-8")), path
        time.sleep(0.01)
    raise RuntimeError(f"missing {operation} trace in {trace_dir}")


def _stage_seconds(stage: Any) -> float:
    return float(stage.get("wall_seconds", stage.get("elapsed_wall_seconds", 0.0))) if isinstance(stage, dict) else 0.0


def _calls(stage: Any) -> list[dict[str, Any]]:
    return stage.get("calls", []) if isinstance(stage, dict) else []


def _localizer_seconds(diagnostics: dict[str, Any], kind: str) -> float:
    return _stage_seconds(diagnostics.get("localization", {}).get(kind, {}))


def _full_stages(payload: dict[str, Any]) -> dict[str, float]:
    diagnostics = payload.get("diagnostics", {})
    pipeline = diagnostics.get("pipeline", {})
    parsing = float(pipeline.get("parsing_validation_seconds", 0.0))
    result_assembly = float(pipeline.get("result_assembly_seconds", 0.0))
    total = float(payload.get("total_seconds", 0.0))
    values = {
        "input_preparation": float(diagnostics.get("benchmark_input_preparation_seconds", 0.0)),
        "document_localization": _localizer_seconds(diagnostics, "docaligner"),
        "mrz_localization": _localizer_seconds(diagnostics, "mrz"),
        "canonicalization": float(pipeline.get("canonicalization_seconds", 0.0)),
        "document_cropping": float(pipeline.get("data_crop_seconds", 0.0)),
        "text_detection": _stage_seconds(diagnostics.get("text_detection", {})),
        "text_line_cropping": float(diagnostics.get("line_crop_seconds", 0.0)),
        "text_recognition": _stage_seconds(diagnostics.get("text_recognition", {})),
        "mrz_recognition": float(pipeline.get("mrz_crop_preprocess_seconds", 0.0)) + _stage_seconds(diagnostics.get("mrz_recognition", {})),
        "ocr_unpacking": float(diagnostics.get("result_unpack_seconds", 0.0)),
        "field_extraction": parsing,
        "result_assembly": max(0.0, result_assembly - parsing),
    }
    values["other"] = max(0.0, total - sum(values.values()))
    values["total"] = total
    return values


def _direct_stages(ocr: dict[str, Any], check: dict[str, Any]) -> dict[str, float]:
    ocr_diag = ocr.get("diagnostics", {})
    values = {
        "input_preparation": float(ocr.get("image_decode_seconds", 0.0)),
        "text_detection": _stage_seconds(ocr_diag.get("text_detection", {})),
        "text_line_cropping": float(ocr_diag.get("line_crop_seconds", 0.0)),
        "text_recognition": _stage_seconds(ocr_diag.get("text_recognition", {})),
        "ocr_unpacking": float(ocr_diag.get("result_unpack_seconds", 0.0)),
        "candidate_construction": float(check.get("token_span_construction_seconds", 0.0)) + float(check.get("candidate_generation_seconds", 0.0)) - float(check.get("geometry_assembly_seconds", 0.0)) - float(check.get("mrz_validation_seconds", 0.0)),
        "geometry_assembly": float(check.get("geometry_assembly_seconds", 0.0)),
        "matching": float(check.get("matching_seconds", 0.0)),
        "mrz_validation": float(check.get("mrz_validation_seconds", 0.0)),
        "strict_field_validation": float(check.get("strict_field_validation_seconds", 0.0)),
        "result_assembly": float(ocr.get("response_assembly_seconds", 0.0)) + float(check.get("response_assembly_seconds", 0.0)),
    }
    total = float(ocr.get("total_server_seconds", 0.0)) + float(check.get("total_server_seconds", 0.0))
    # Candidate/geometry/MRZ timings are nested inside matching; keep them
    # visible for diagnosis but exclude them from the top-level residual.
    top_level = ("input_preparation", "text_detection", "text_line_cropping", "text_recognition", "ocr_unpacking", "matching", "result_assembly")
    values["other"] = max(0.0, total - sum(values[name] for name in top_level))
    values["total"] = total
    return values


def _workload(candidate: str, document: Document, diagnostics: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    stages = diagnostics.get("diagnostics", diagnostics)
    detection = stages.get("text_detection", {})
    recognition = stages.get("text_recognition", {})
    line_filter = stages.get("line_filter", {})
    calls = _calls(detection)
    detector_shapes = [call.get("tensor_shapes", call.get("submitted_input_shapes", [])) for call in calls]
    detector_sizes = [call.get("detector_resized_shapes", []) for call in calls]
    pixels = sum(int(call.get("padded_tensor_pixel_area") or 0) for call in calls)
    if not pixels:
        pixels = sum(sum(int(item[-2]) * int(item[-1]) for item in call.get("tensor_shapes", []) if isinstance(item, (list, tuple)) and len(item) >= 2) for call in calls)
    line_count = int(line_filter.get("detected_line_count", 0))
    candidates = int(line_filter.get("recognition_candidate_count", 0))
    recognition_seconds = _stage_seconds(recognition)
    detection_row = {
        "candidate": candidate, "document_type": document.document_type, "document_id": document.document_id,
        "detector_images_per_document": sum(call.get("submitted_batch_size", 0) for call in calls),
        "input_widths": [call.get("input_widths", []) for call in calls], "input_heights": [call.get("input_heights", []) for call in calls],
        "effective_resized_dimensions": detector_sizes, "padded_tensor_dimensions": detector_shapes,
        "total_detector_pixels": pixels, "detection_inference_calls": len(calls),
        "actual_detection_tensor_batches": [call.get("tensor_batch_sizes", [call.get("tensor_batch_size")]) for call in calls],
        "detected_line_count": line_count, "filtered_before_recognition_count": int(line_filter.get("filtered_before_recognition_count", 0)),
    }
    crops = [record for call in _calls(recognition) for record in call.get("crop_records", [])]
    widths = [record.get("original_crop_w") for record in crops if record.get("original_crop_w") is not None]
    heights = [record.get("original_crop_h") for record in crops if record.get("original_crop_h") is not None]
    buckets = {name: 0 for name in ("<64", "64-127", "128-255", "256-511", "512+")}
    for width in widths:
        buckets["<64" if width < 64 else "64-127" if width < 128 else "128-255" if width < 256 else "256-511" if width < 512 else "512+"] += 1
    recognition_row = {
        "candidate": candidate, "document_type": document.document_type, "document_id": document.document_id,
        "recognition_crops_per_document": candidates, "crop_widths": widths, "crop_heights": heights,
        "width_bucket_distribution": buckets, "recognition_candidate_count": candidates,
        "recognition_inference_calls": len(_calls(recognition)), "actual_recognition_tensor_batches": [call.get("tensor_batch_sizes", [call.get("tensor_batch_size")]) for call in _calls(recognition)],
        "recognition_lines_per_second": candidates / recognition_seconds if recognition_seconds else None,
    }
    return detection_row, recognition_row


class Server:
    def __init__(self, args: argparse.Namespace, directory: Path, candidate: str, env: dict[str, str]):
        self.args, self.directory, self.candidate, self.env = args, directory, candidate, env
        self.process: subprocess.Popen[str] | None = None
        self.peak_rss = 0

    def _sample(self) -> None:
        value = _rss(self.process.pid if self.process else None)
        if value is not None:
            self.peak_rss = max(self.peak_rss, value)

    def start(self) -> dict[str, Any]:
        if _port_open(self.args.port):
            raise RuntimeError(f"port {self.args.port} is already listening")
        trace_dir = self.directory / "trace"
        trace_dir.mkdir(parents=True, exist_ok=True)
        started = time.perf_counter()
        environment = os.environ.copy()
        environment.update(self.env, MODEL_DIR=str(self.args.model_dir.resolve()), VOIGHT_BENCHMARK_TRACE_DIR=str(trace_dir))
        log = (self.directory / "server.log").open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(self.args.port), "--workers", "1"],
            cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT, start_new_session=True, text=True,
        )
        log.close()
        deadline = time.monotonic() + self.args.timeout
        while time.monotonic() < deadline:
            self._sample()
            if self.process.poll() is not None:
                raise RuntimeError(f"server exited with code {self.process.returncode}; see {self.directory / 'server.log'}")
            try:
                response = requests.get(f"http://127.0.0.1:{self.args.port}/v1/health/live", timeout=5)
                if response.ok:
                    return {"live_seconds": time.perf_counter() - started, "status": response.json()}
            except requests.RequestException:
                pass
            time.sleep(0.2)
        raise RuntimeError(f"server did not become live; see {self.directory / 'server.log'}")

    def stop(self) -> dict[str, Any]:
        if self.process is None:
            return {"peak_rss_mb": None, "cleanup_verified": True}
        pid = self.process.pid
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            return {"peak_rss_mb": round(self.peak_rss / 1024 / 1024, 3), "cleanup_verified": True}
        self.process.send_signal(signal.SIGTERM)
        try:
            self.process.wait(timeout=45)
        except subprocess.TimeoutExpired:
            os.killpg(pgid, signal.SIGKILL)
            self.process.wait(timeout=15)
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline and (_port_open(self.args.port) or _pids_in_group(pgid)):
            time.sleep(0.2)
        remaining = _pids_in_group(pgid)
        return {
            "peak_rss_mb": round(self.peak_rss / 1024 / 1024, 3), "server_pid": pid, "server_pgid": pgid,
            "returncode": self.process.returncode, "remaining_pids": remaining,
            "cleanup_verified": not remaining and not _port_open(self.args.port),
        }


def _request(server: Server, path: str, files: Any, json_body: dict[str, Any] | None = None) -> tuple[dict[str, Any], float]:
    started = time.perf_counter()
    response = requests.post(f"http://127.0.0.1:{server.args.port}{path}", files=files if json_body is None else None, json=json_body, timeout=server.args.timeout)
    elapsed = time.perf_counter() - started
    if not response.ok:
        raise RuntimeError(f"{path} failed {response.status_code}: {response.text[:1000]}")
    return response.json(), elapsed


def _full_request(server: Server, document: Document) -> tuple[dict[str, Any], dict[str, Any]]:
    kind, _ = ROUTES[document.document_type]
    files = [("images", (path.name, path.read_bytes(), _mime(path))) for _, path in document.paths]
    if document.document_type == "id_card":
        import io
        import zipfile
        body = io.BytesIO()
        with zipfile.ZipFile(body, "w") as archive:
            for role, path in document.paths:
                archive.writestr(f"card/{role}{path.suffix.lower()}", path.read_bytes())
        files = {"archive": ("card.zip", body.getvalue(), "application/zip")}
    payload, client = _request(server, f"/v1/ocr/{kind}/batch", files)
    item = payload["items"][0]
    if not item.get("success"):
        raise RuntimeError(f"full item failed: {item.get('error')}")
    result = item["result"]
    return {"payload": payload, "result": result, "client_seconds": client}, payload


def _direct_request(server: Server, document: Document) -> tuple[dict[str, Any], dict[str, Any]]:
    _, kind = ROUTES[document.document_type]
    uploaded = _files(document)
    ocr_payload, ocr_client = _request(server, f"/verification/{kind}/ocr", uploaded)
    before = set((server.directory / "trace").glob("check-*.json"))
    check_payload, check_client = _request(server, f"/verification/{kind}/check", None, json_body={"ocr": ocr_payload, "fields": _truth(document)})
    check_trace, check_path = _trace(server.directory / "trace", "check", before)
    before_ocr = set((server.directory / "trace").glob("ocr-*.json"))
    # The OCR trace was created before the check. Select the newest trace at this point.
    ocr_trace, ocr_path = _trace(server.directory / "trace", "ocr", before_ocr)
    # If the trace existed before this call, the fallback below finds its newest file.
    if not ocr_trace:
        ocr_path = max((server.directory / "trace").glob("ocr-*.json"), key=lambda item: item.stat().st_mtime_ns)
        ocr_trace = json.loads(ocr_path.read_text(encoding="utf-8"))
    return {
        "ocr_payload": ocr_payload, "check_payload": check_payload,
        "ocr_trace": ocr_trace, "check_trace": check_trace,
        "ocr_client_seconds": ocr_client, "check_client_seconds": check_client,
        "ocr_trace_path": str(ocr_path), "check_trace_path": str(check_path),
    }, check_payload


def _direct_request_fixed(server: Server, document: Document) -> tuple[dict[str, Any], dict[str, Any]]:
    _, kind = ROUTES[document.document_type]
    before_ocr = set((server.directory / "trace").glob("ocr-*.json"))
    ocr_payload, ocr_client = _request(server, f"/verification/{kind}/ocr", _files(document))
    ocr_trace, ocr_path = _trace(server.directory / "trace", "ocr", before_ocr)
    before_check = set((server.directory / "trace").glob("check-*.json"))
    check_payload, check_client = _request(server, f"/verification/{kind}/check", None, json_body={"ocr": ocr_payload, "fields": _truth(document)})
    check_trace, check_path = _trace(server.directory / "trace", "check", before_check)
    return {"ocr_payload": ocr_payload, "check_payload": check_payload, "ocr_trace": ocr_trace, "check_trace": check_trace, "ocr_client_seconds": ocr_client, "check_client_seconds": check_client, "ocr_trace_path": str(ocr_path), "check_trace_path": str(check_path)}, check_payload


def _run_document(server: Server, candidate: str, document: Document) -> dict[str, Any]:
    if candidate == FULL:
        data, _ = _full_request(server, document)
        payload = data["payload"]
        result = data["result"]
        stages = _full_stages(payload)
        fields = {name: entry.get("value") for name, entry in result.get("fields", {}).items()}
        matcher = {}
        diagnostics = payload.get("diagnostics", {})
        detection, recognition = _workload(candidate, document, diagnostics)
        return {"candidate": candidate, "document_type": document.document_type, "document_id": document.document_id, "client_seconds": data["client_seconds"], "server_seconds": float(payload.get("total_seconds", stages["total"])), "stages": stages, "fields": fields, "field_raw": {name: entry.get("raw_text", []) for name, entry in result.get("fields", {}).items()}, "matcher": matcher, "ocr": diagnostics, "detection": detection, "recognition": recognition, "raw_result": result}
    data, _ = _direct_request_fixed(server, document)
    ocr_trace, check_trace = data["ocr_trace"], data["check_trace"]
    check = data["check_payload"]
    stages = _direct_stages(ocr_trace, check_trace)
    fields = {name: entry.get("detected") for name, entry in check.get("fields", {}).items()}
    matcher = {name: {"status": entry.get("status"), "score": entry.get("score"), "source": entry.get("source"), "evidence": entry.get("evidence", [])} for name, entry in check.get("fields", {}).items()}
    diagnostics = ocr_trace.get("diagnostics", {})
    detection, recognition = _workload(candidate, document, diagnostics)
    return {"candidate": candidate, "document_type": document.document_type, "document_id": document.document_id, "client_seconds": data["ocr_client_seconds"] + data["check_client_seconds"], "server_seconds": stages["total"], "stages": stages, "fields": fields, "field_raw": {}, "matcher": matcher, "ocr": diagnostics, "detection": detection, "recognition": recognition, "raw_ocr": data["ocr_payload"], "check_trace": check_trace}


def _warmup(server: Server, candidate: str, documents: list[Document]) -> float:
    started = time.perf_counter()
    for document in documents:
        _run_document(server, candidate, document)
    return time.perf_counter() - started


def _percentile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    if not values:
        return 0.0
    position = (len(values) - 1) * fraction
    lower, upper = int(position), min(len(values) - 1, int(position) + 1)
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _stats(values: list[float]) -> dict[str, float]:
    return {"median": statistics.median(values), "mean": statistics.mean(values), "p50": _percentile(values, .50), "p90": _percentile(values, .90), "p95": _percentile(values, .95), "minimum": min(values), "maximum": max(values), "standard_deviation": statistics.stdev(values) if len(values) > 1 else 0.0, "docs_per_sec": 1 / statistics.mean(values) if statistics.mean(values) else 0.0}


def _config(candidate: str, model_dir: Path) -> dict[str, Any]:
    common = {**BASE_ENV, "MODEL_DIR": str(model_dir.resolve()), "CPU_ONLY": True, "packing_strategy": "fixed-width"}
    if candidate == FULL:
        return {"name": candidate, "route": "/v1/ocr/{type}/batch", "architecture": {"document_localization": True, "canonicalization": True, "document_cropping": True, "new_matcher": False}, "models": {"detector": DETECTOR, "recognizer": RECOGNIZER, "recognizer_backend": "paddle", "localization_model": "fastvit_sa24", "mrz_localizer": "20250222", "mrz_backend": "generic-paddle"}, "batch": {"localization": 4, "detection": 1, "recognition": 2, "mrz_recognition": 2}, "verification_overrides": {}, "environment": common}
    return {"name": candidate, "route": "/verification/{type}/ocr + /verification/{type}/check", "architecture": {"document_localization": False, "canonicalization": False, "document_cropping": False, "new_matcher": True}, "models": {"detector": DETECTOR, "recognizer": RECOGNIZER, "recognizer_backend": "paddle", "localization_model": "not part of route", "mrz_localizer": "not part of route", "mrz_backend": "validated candidates from direct OCR only"}, "batch": {"localization": "N/A", "detection": 1, "recognition": 4, "mrz_recognition": "N/A"}, "verification_overrides": {"VERIFICATION_TEXT_DETECTION_BATCH_SIZE": 1, "VERIFICATION_TEXT_RECOGNITION_BATCH_SIZE": 4}, "environment": {**common, "VERIFICATION_TEXT_DETECTION_BATCH_SIZE": "1", "VERIFICATION_TEXT_RECOGNITION_BATCH_SIZE": "4"}}


def _evaluate(documents: list[Document], runs: list[dict[str, Any]], candidate: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    fields = []
    documents_rows = []
    for run in runs:
        document = next(doc for doc in documents if doc.document_id == run["document_id"])
        for field, expected in _truth(document).items():
            detected = run["fields"].get(field)
            status = _field_status(expected, detected, field)
            matcher = run.get("matcher", {}).get(field, {})
            fields.append({"candidate": candidate, "document_type": document.document_type, "document_id": document.document_id, "field": field, "expected": expected, "detected": detected, "neutral_status": status, "neutral_correct": status == "correct", "matcher_status": matcher.get("status"), "matcher_accepted": matcher.get("status") in {"match", "likely_match"}, "score": matcher.get("score"), "source": matcher.get("source"), "evidence": matcher.get("evidence", [])})
        doc_fields = [row for row in fields if row["candidate"] == candidate and row["document_id"] == document.document_id]
        documents_rows.append({"candidate": candidate, "document_type": document.document_type, "document_id": document.document_id, "fields": len(doc_fields), "correct": sum(row["neutral_correct"] for row in doc_fields), "incorrect": sum(row["neutral_status"] == "incorrect" for row in doc_fields), "not_found": sum(row["neutral_status"] == "not_found" for row in doc_fields), "hard_failures": sum(row["neutral_status"] != "correct" for row in doc_fields), "perfect": all(row["neutral_status"] == "correct" for row in doc_fields), "zero_hard_failure": all(row["neutral_status"] == "correct" for row in doc_fields)})
    return fields, documents_rows, [row for row in fields if row["candidate"] == candidate]


def _accuracy_rows(documents: list[Document], fields: list[dict[str, Any]], docs: list[dict[str, Any]], candidate: str) -> list[dict[str, Any]]:
    result = []
    groups = [(kind, [doc for doc in documents if doc.document_type == kind]) for kind in KINDS] + [("overall", documents)]
    for kind, selected in groups:
        f = [row for row in fields if kind == "overall" or row["document_type"] == kind]
        d = [row for row in docs if kind == "overall" or row["document_type"] == kind]
        result.append({"candidate": candidate, "document_type": kind, "unique_evaluated_fields": len(f), "correct_accepted": sum(row["neutral_correct"] for row in f), "accuracy": sum(row["neutral_correct"] for row in f) / len(f), "incorrect": sum(row["neutral_status"] == "incorrect" for row in f), "not_found": sum(row["neutral_status"] == "not_found" for row in f), "hard_failures": sum(row["neutral_status"] != "correct" for row in f), "hard_failure_rate": sum(row["neutral_status"] != "correct" for row in f) / len(f), "documents": len(selected), "perfect_documents": sum(row["perfect"] for row in d), "perfect_document_rate": sum(row["perfect"] for row in d) / len(d), "zero_hard_failure_documents": sum(row["zero_hard_failure"] for row in d), "zero_hard_failure_document_rate": sum(row["zero_hard_failure"] for row in d) / len(d), "matcher_accepted": sum(row["matcher_accepted"] for row in f), "matcher_likely_match": sum(row["matcher_status"] == "likely_match" for row in f)})
    return result


def _strict_rows(fields: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**row, "strict_field": True, "correct_accepted": row["neutral_correct"], "incorrect_accepted": row["matcher_accepted"] and not row["neutral_correct"], "mismatch": row["neutral_status"] == "incorrect", "not_found": row["neutral_status"] == "not_found"} for row in fields if row["field"] in STRICT_FIELDS]


def _known_failures(documents: list[Document], field_rows: list[dict[str, Any]], runs_by_key: dict[tuple[str, str], dict[str, Any]]) -> list[dict[str, Any]]:
    keys = {("p_3", "type"), ("p_7", "name"), ("p_9", "passport_number"), ("d_3", "surname"), ("d_6", "expiry_date"), ("d_6", "personal_id"), ("d_6", "serial_number")}
    rows = []
    for document_id, field in sorted(keys):
        doc = next(doc for doc in documents if doc.document_id == document_id)
        expected = _truth(doc).get(field, "")
        full = next((row for row in field_rows if row["candidate"] == FULL and row["document_id"] == document_id and row["field"] == field), {})
        direct = next((row for row in field_rows if row["candidate"] == DIRECT and row["document_id"] == document_id and row["field"] == field), {})
        full_run, direct_run = runs_by_key.get((FULL, document_id), {}), runs_by_key.get((DIRECT, document_id), {})
        if full.get("neutral_status") == "correct" and direct.get("neutral_status") != "correct": cause = "direct-image geometry/association"
        elif direct.get("neutral_status") == "correct" and full.get("neutral_status") != "correct": cause = "document localization/canonical crop"
        elif field == "surname" and document_id == "d_3" and direct.get("neutral_status") == "not_found": cause = "annotation issue / no usable evidence"
        else: cause = "text recognition or image quality (shared model evidence)"
        rows.append({"document_type": doc.document_type, "document_id": document_id, "field": field, "annotation": expected, "FULL_LATIN_PIPELINE_output": full.get("detected"), "DIRECT_MATCHING_PIPELINE_output": direct.get("detected"), "FULL_status": full.get("neutral_status"), "DIRECT_status": direct.get("neutral_status"), "FULL_raw_evidence": full_run.get("field_raw", {}).get(field, []), "DIRECT_raw_OCR_evidence": direct_run.get("raw_ocr", {}), "failure_cause": cause})
    return rows


def _model_sizes(model_dir: Path, names: list[str]) -> dict[str, int]:
    result = {}
    for name in names:
        root = model_dir / "official_models" / name
        result[name] = sum(path.stat().st_size for path in root.rglob("*") if path.is_file()) if root.is_dir() else 0
    return result


def _report(output: Path, accuracy: list[dict[str, Any]], performance: list[dict[str, Any]], stage_rows: list[dict[str, Any]], transitions: list[dict[str, Any]], strict: list[dict[str, Any]], detection: list[dict[str, Any]], recognition: list[dict[str, Any]], resources: list[dict[str, Any]], known: list[dict[str, Any]], configs: dict[str, Any]) -> None:
    overall = {row["candidate"]: row for row in accuracy if row["document_type"] == "overall"}
    perf = {row["candidate"]: row for row in performance}
    stage = {(row["candidate"], row["stage"]): row["median_ms_per_doc"] for row in stage_rows}
    def v(candidate: str, name: str) -> str:
        value = stage.get((candidate, name))
        return "N/A" if value is None else f"{value:.2f}"
    transitions_count = {name: sum(row["transition"] == name for row in transitions) for name in sorted({row["transition"] for row in transitions})}
    full_only = sum(row["transition"] == "FULL_correct->DIRECT_failure" for row in transitions)
    direct_only = sum(row["transition"] == "FULL_failure->DIRECT_correct" for row in transitions)
    both = sum(row["transition"] == "both_correct" for row in transitions)
    fail = sum(row["transition"] == "both_fail" for row in transitions)
    lines = [
        "# Full Pipeline vs Direct Matching", "", "## Executive Summary", "",
        "CPU-only, five measured passes after one full-dataset warmup per fresh candidate process. Primary accuracy is neutral normalized field equality against the frozen annotation truth; DIRECT matcher status is retained separately and never substitutes for the common score.", "",
        f"- FULL_LATIN_PIPELINE: `{overall[FULL]['correct_accepted']}/{overall[FULL]['unique_evaluated_fields']}` (`{overall[FULL]['accuracy']:.2%}`), `{overall[FULL]['hard_failures']}` hard failures, `{overall[FULL]['zero_hard_failure_documents']}/20` zero-hard-failure docs, `{overall[FULL]['perfect_documents']}/20` perfect docs.",
        f"- DIRECT_MATCHING_PIPELINE: `{overall[DIRECT]['correct_accepted']}/{overall[DIRECT]['unique_evaluated_fields']}` (`{overall[DIRECT]['accuracy']:.2%}`), `{overall[DIRECT]['hard_failures']}` hard failures, `{overall[DIRECT]['zero_hard_failure_documents']}/20` zero-hard-failure docs, `{overall[DIRECT]['perfect_documents']}/20` perfect docs; matcher accepted `{overall[DIRECT]['matcher_accepted']}/254` including `{overall[DIRECT]['matcher_likely_match']}` likely matches.",
        "", "## Exact Configurations", "", "| Setting | FULL_LATIN_PIPELINE | DIRECT_MATCHING_PIPELINE |", "|---|---|---|",
        "| Localization / canonicalization / document crop | yes / yes / yes | no / no / no |", f"| Detector | `{DETECTOR}` | `{DETECTOR}` |", f"| Recognizer | `{RECOGNIZER}` / Paddle | `{RECOGNIZER}` / Paddle |", "| Batches | localization 4; detection 1; recognition 2; MRZ 2 | detection 1; verification recognition 4 |", "| Packing | fixed-width | fixed-width |", "| New matcher | no | yes |", "",
        "## Accuracy", "", "| Candidate | Fields | Correct/accepted | Accuracy | Hard failures | Zero-HF docs | Perfect docs |", "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for candidate in (FULL, DIRECT):
        row = overall[candidate]; lines.append(f"| {candidate} | {row['unique_evaluated_fields']} | {row['correct_accepted']} | {row['accuracy']:.2%} | {row['hard_failures']} ({row['hard_failure_rate']:.2%}) | {row['zero_hard_failure_documents']}/20 ({row['zero_hard_failure_document_rate']:.2%}) | {row['perfect_documents']}/20 ({row['perfect_document_rate']:.2%}) |")
    lines += ["", "### By document type", "", "| Type | FULL accuracy | DIRECT accuracy | FULL hard failures | DIRECT hard failures |", "|---|---:|---:|---:|---:|"]
    for kind in KINDS:
        rows = {(row["candidate"], row["document_type"]): row for row in accuracy}; lines.append(f"| {kind} | {rows[FULL, kind]['accuracy']:.2%} | {rows[DIRECT, kind]['accuracy']:.2%} | {rows[FULL, kind]['hard_failures']} | {rows[DIRECT, kind]['hard_failures']} |")
    lines += ["", "### Field transitions", "", f"- FULL-only successes: `{full_only}`; DIRECT-only successes: `{direct_only}`; both succeed: `{both}`; both fail: `{fail}`.", f"- Transition counts: `{_json(transitions_count)}`.", "- Every transition is in `field_transitions.csv`; known cases are in `known_failure_comparison.csv`.", "", "### Strict identifiers/dates", ""]
    for candidate in (FULL, DIRECT):
        rows = [row for row in strict if row["candidate"] == candidate]; lines.append(f"- {candidate}: correct accepted `{sum(row['correct_accepted'] for row in rows)}`, incorrect accepted `{sum(row['incorrect_accepted'] for row in rows)}`, mismatch `{sum(row['mismatch'] for row in rows)}`, not found `{sum(row['not_found'] for row in rows)}`.")
    lines += ["", "## Known Failures", "", "See `known_failure_comparison.csv` for annotation, both outputs, raw evidence, and cause for all seven requested cases.", "", "## Performance", "", "| Candidate | Median ms/doc | Mean | P50 | P90 | P95 | Min | Max | Std dev | Docs/sec |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for candidate in (FULL, DIRECT):
        row = perf[candidate]; lines.append(f"| {candidate} | {row['median_ms_per_doc']:.2f} | {row['mean_ms_per_doc']:.2f} | {row['p50_ms_per_doc']:.2f} | {row['p90_ms_per_doc']:.2f} | {row['p95_ms_per_doc']:.2f} | {row['minimum_ms_per_doc']:.2f} | {row['maximum_ms_per_doc']:.2f} | {row['standard_deviation_ms_per_doc']:.2f} | {row['docs_per_sec']:.4f} |")
    faster = FULL if perf[FULL]["median_ms_per_doc"] < perf[DIRECT]["median_ms_per_doc"] else DIRECT
    delta = perf[DIRECT]["median_ms_per_doc"] - perf[FULL]["median_ms_per_doc"]
    lines += ["", "## Timing Breakdown", "", "| Stage | FULL | DIRECT | Delta (DIRECT-FULL) |", "|---|---:|---:|---:|"]
    stages = [("Input preparation", "input_preparation"), ("Document localization", "document_localization"), ("MRZ localization", "mrz_localization"), ("Canonicalization", "canonicalization"), ("Document cropping", "document_cropping"), ("Text detection", "text_detection"), ("Text-line cropping", "text_line_cropping"), ("Text recognition", "text_recognition"), ("MRZ work", "mrz_recognition"), ("OCR unpacking", "ocr_unpacking"), ("Candidate construction", "candidate_construction"), ("Geometry assembly", "geometry_assembly"), ("Matching / field extraction", "matching"), ("Result assembly", "result_assembly"), ("Other", "other"), ("TOTAL", "total")]
    for label, name in stages:
        left, right = stage.get((FULL, name)), stage.get((DIRECT, name)); delta_text = "N/A" if left is None or right is None else f"{right-left:.2f}"
        lines.append(f"| {label} | {('N/A' if left is None else f'{left:.2f}')} | {('N/A' if right is None else f'{right:.2f}')} | {delta_text} |")
    lines += ["", "## OCR Workload", "", "Detection and recognition workload rows retain actual calls, tensor batches, input dimensions, pixels, crop dimensions, and width buckets. Full localization/cropping can reduce the downstream image size; direct OCR sees the raw source images.", "", "## Memory / Resident Models", "", "See `resource_usage.csv`; RSS is CPU process RSS, not VRAM. Model file sizes are on-disk cache sizes, not RAM usage.", "", "## Why One Is Faster", "", f"FULL localization + canonicalization + cropping median cost is `{(stage.get((FULL, 'document_localization'), 0) + stage.get((FULL, 'mrz_localization'), 0) + stage.get((FULL, 'canonicalization'), 0) + stage.get((FULL, 'document_cropping'), 0)):.2f} ms/doc`. Its downstream detection + recognition + MRZ work is `{(stage.get((FULL, 'text_detection'), 0) + stage.get((FULL, 'text_recognition'), 0) + stage.get((FULL, 'mrz_recognition'), 0)):.2f} ms/doc`; DIRECT downstream detection + recognition + matching is `{(stage.get((DIRECT, 'text_detection'), 0) + stage.get((DIRECT, 'text_recognition'), 0) + stage.get((DIRECT, 'matching'), 0)):.2f} ms/doc`.", f"The measured median winner is `{faster}` by `{abs(delta):.2f} ms/doc` (`{abs(delta)/min(perf[FULL]['median_ms_per_doc'], perf[DIRECT]['median_ms_per_doc']):.2%}` relative to the faster value).", "", "## Recommendation", "", f"Accuracy winner under the neutral evaluator: `{FULL if overall[FULL]['accuracy'] > overall[DIRECT]['accuracy'] else DIRECT if overall[DIRECT]['accuracy'] > overall[FULL]['accuracy'] else 'tie'}`. Latency winner: `{faster}`. Keep both only if the product needs direct verification's raw-image fallback or its matcher-specific provenance/status contract; this benchmark does not justify merging the routes.", ""]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", ".paddlex")))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8044)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()
    if not args.model_dir.is_dir():
        raise SystemExit(f"missing model directory: {args.model_dir}")
    for model in (DETECTOR, RECOGNIZER):
        if not (args.model_dir / "official_models" / model / "inference.pdiparams").is_file():
            raise SystemExit(f"missing required model asset: {model}")
    documents, manifest = validate_and_manifest(ROOT / "dataset")
    if len(documents) != 20 or manifest["physical_images"] != 24 or sum(len(_truth(doc)) for doc in documents) != 254:
        raise SystemExit("frozen dataset must be 20 logical documents, 24 physical images, and 254 value fields")
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=False)
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    configs = {FULL: _config(FULL, args.model_dir), DIRECT: _config(DIRECT, args.model_dir)}
    for candidate, config in configs.items():
        (output / ("full_pipeline_config.json" if candidate == FULL else "direct_matching_config.json")).write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")
    environment = {"cpu_only": True, "dataset": {"logical_documents": len(documents), "physical_images": manifest["physical_images"], "unique_evaluated_fields": 254, "manifest_sha256": hashlib.sha256(_json(manifest).encode()).hexdigest()}, "python": sys.version, "platform": platform.platform(), "cpu": platform.processor(), "model_dir": str(args.model_dir.resolve()), "warmup_passes": 1, "measured_passes": 5, "execution_order": [FULL, DIRECT, DIRECT, FULL, FULL, DIRECT, DIRECT, FULL, FULL, DIRECT], "candidates": configs, "prior_misleading_artifact": "outputs/benchmarks/22.latin-vs-current-matcher/20260902T112500Z was old matcher vs new matcher and is not used as result evidence"}
    (output / "environment.json").write_text(json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8")

    raw_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    detection_rows: list[dict[str, Any]] = []
    recognition_rows: list[dict[str, Any]] = []
    resource_rows: list[dict[str, Any]] = []
    all_runs: dict[str, list[dict[str, Any]]] = {FULL: [], DIRECT: []}
    order = [FULL, DIRECT, DIRECT, FULL, FULL, DIRECT, DIRECT, FULL, FULL, DIRECT]
    for sequence, candidate in enumerate(order, 1):
        pass_number = sum(item == candidate for item in order[:sequence])
        label = "full-latin-pipeline" if candidate == FULL else "direct-matching-pipeline"
        directory = output / f"{sequence:02d}.{label}-pass-{pass_number}"
        directory.mkdir()
        env = {**BASE_ENV, **({"VERIFICATION_TEXT_DETECTION_BATCH_SIZE": "1", "VERIFICATION_TEXT_RECOGNITION_BATCH_SIZE": "4"} if candidate == DIRECT else {})}
        server = Server(args, directory, candidate, env)
        started = time.perf_counter()
        server.start()
        warmup_seconds = _warmup(server, candidate, documents)
        measured_started = time.perf_counter()
        for document in documents:
            run = _run_document(server, candidate, document)
            run.update({"pass": pass_number, "sequence": sequence, "warmup_seconds": warmup_seconds, "measured_wall_seconds": time.perf_counter() - measured_started})
            all_runs[candidate].append(run)
            raw_rows.append({"candidate": candidate, "pass": pass_number, "sequence": sequence, "document_type": document.document_type, "document_id": document.document_id, "client_ms": run["client_seconds"] * 1000, "server_ms": run["server_seconds"] * 1000, "stages_ms": {name: value * 1000 for name, value in run["stages"].items()}, "fields": run["fields"], "matcher": run.get("matcher", {}), "trace": run.get("check_trace", {})})
            detection_rows.append({"pass": pass_number, **run["detection"]})
            recognition_rows.append({"pass": pass_number, **run["recognition"]})
        resources = server.stop()
        loads = []
        for path in sorted((directory / "trace").glob("model-load-*.json")):
            loads.append(json.loads(path.read_text(encoding="utf-8")))
        (directory / "model_loads.json").write_text(json.dumps(loads, indent=2), encoding="utf-8")
        resource_rows.append({"candidate": candidate, "pass": pass_number, "peak_rss_mb": resources.get("peak_rss_mb"), "warmup_wall_seconds": warmup_seconds, "model_initialization_seconds": sum(float(load.get("seconds", 0.0)) for load in loads), "resident_models": sorted({load.get("model") for load in loads}), "model_loads": loads, "model_file_sizes_bytes": _model_sizes(args.model_dir, [DETECTOR, RECOGNIZER]), "cleanup_verified": resources.get("cleanup_verified")})
        if not resources.get("cleanup_verified"):
            raise RuntimeError(f"server cleanup failed for {candidate} pass {pass_number}")
        print(f"completed {candidate} pass {pass_number}/5 ({time.perf_counter() - started:.1f}s)", flush=True)

    all_field_rows: list[dict[str, Any]] = []
    all_doc_rows: list[dict[str, Any]] = []
    accuracy_rows: list[dict[str, Any]] = []
    for candidate in (FULL, DIRECT):
        fields, docs, _ = _evaluate(documents, [run for run in all_runs[candidate] if run["pass"] == 1], candidate)
        all_field_rows.extend(fields); all_doc_rows.extend(docs); accuracy_rows.extend(_accuracy_rows(documents, fields, docs, candidate))
    by_key = {(row["candidate"], row["document_id"]): row for row in all_field_rows}
    transitions = []
    for document in documents:
        for field in _truth(document):
            left, right = by_key[FULL, document.document_id], by_key[DIRECT, document.document_id]
            left = next(row for row in all_field_rows if row["candidate"] == FULL and row["document_id"] == document.document_id and row["field"] == field)
            right = next(row for row in all_field_rows if row["candidate"] == DIRECT and row["document_id"] == document.document_id and row["field"] == field)
            left_ok, right_ok = left["neutral_correct"], right["neutral_correct"]
            transitions.append({"document_type": document.document_type, "document_id": document.document_id, "field": field, "FULL_status": left["neutral_status"], "DIRECT_status": right["neutral_status"], "FULL_output": left["detected"], "DIRECT_output": right["detected"], "transition": "both_correct" if left_ok and right_ok else "FULL_correct->DIRECT_failure" if left_ok else "FULL_failure->DIRECT_correct" if right_ok else "both_fail"})
    strict = _strict_rows(all_field_rows)
    runs_by_key = {(run["candidate"], run["document_id"]): run for candidate in (FULL, DIRECT) for run in all_runs[candidate][:20]}
    known = _known_failures(documents, all_field_rows, runs_by_key)
    performance = []
    for candidate in (FULL, DIRECT):
        values = [run["server_seconds"] for run in all_runs[candidate]]
        stats = _stats(values)
        performance.append({"candidate": candidate, "documents": len(values), **{f"{key}_ms_per_doc": value * 1000 for key, value in stats.items() if key != "docs_per_sec"}, "docs_per_sec": stats["docs_per_sec"], "client_mean_ms_per_doc": statistics.mean(run["client_seconds"] for run in all_runs[candidate]) * 1000})
        for stage in sorted({name for run in all_runs[candidate] for name in run["stages"]}):
            values = [run["stages"].get(stage, 0.0) for run in all_runs[candidate]]
            stage_rows.append({"candidate": candidate, "stage": stage, "median_ms_per_doc": statistics.median(values) * 1000, "mean_ms_per_doc": statistics.mean(values) * 1000})

    write_csv(output / "raw_runs.csv", raw_rows)
    write_csv(output / "overall_accuracy.csv", [row for row in accuracy_rows if row["document_type"] == "overall"])
    write_csv(output / "document_type_accuracy.csv", [row for row in accuracy_rows if row["document_type"] != "overall"])
    write_csv(output / "field_comparison.csv", all_field_rows)
    write_csv(output / "document_comparison.csv", all_doc_rows)
    write_csv(output / "field_transitions.csv", transitions)
    write_csv(output / "strict_field_results.csv", strict)
    write_csv(output / "stage_timings.csv", stage_rows)
    write_csv(output / "performance.csv", performance)
    write_csv(output / "detection_workload.csv", detection_rows)
    write_csv(output / "recognition_workload.csv", recognition_rows)
    write_csv(output / "resource_usage.csv", resource_rows)
    write_csv(output / "known_failure_comparison.csv", known)
    summary = {"accuracy": accuracy_rows, "performance": performance, "field_transitions": {name: sum(row["transition"] == name for row in transitions) for name in sorted({row["transition"] for row in transitions})}, "strict_incorrect_accepted": {candidate: sum(row["candidate"] == candidate and row["incorrect_accepted"] for row in strict) for candidate in (FULL, DIRECT)}, "known_failures": known, "configs": configs}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    _report(output, accuracy_rows, performance, stage_rows, transitions, strict, detection_rows, recognition_rows, resource_rows, known, configs)
    print(f"completed benchmark: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
