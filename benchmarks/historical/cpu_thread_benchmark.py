"""Fresh-container CPU thread benchmark for the finalized Voight pipeline."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import statistics
import subprocess
import sys
import time
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "outputs/benchmarks/11.cpu-thread-count-benchmark"
IMAGE = "voight:cpu"
PORT = 18080
THREADS = (1, 2, 4, 6, 8, 12, 16)
DOCS = ("passport", "id-card", "driving-license")
DATASET = ROOT / "dataset"
EXPECTED_DOCUMENTS = {"passport": 9, "id-card": 4, "driving-license": 7}
EXPECTED_IMAGES = {"passport": 9, "id-card": 8, "driving-license": 7}
REPEATS = 3
TIMEOUT = 900.0
ENV_KEYS = (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "PADDLE_NUM_THREADS", "CPU_THREADS",
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def run(command: list[str], *, check: bool = True, timeout: float = 120.0) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=check, timeout=timeout)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def host_cpus() -> dict[str, Any]:
    logical = os.cpu_count()
    physical = None
    try:
        rows = run(["lscpu", "-p=CPU,CORE,SOCKET"]).stdout.splitlines()
        physical = len({tuple(row.split(",")[1:]) for row in rows if row and not row.startswith("#")})
    except (OSError, subprocess.SubprocessError):
        pass
    return {"physical_cpu_count": physical, "logical_cpu_count": logical, "platform": platform.platform(), "processor": platform.processor()}


def available_memory_kib() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def container_pid(name: str) -> int | None:
    result = run(["docker", "inspect", "--format", "{{.State.Pid}}", name], check=False)
    if result.returncode or not result.stdout.strip():
        return None
    return int(result.stdout.strip())


def proc_cpu(pid: int) -> tuple[int, int] | None:
    try:
        total = sum(int(value) for value in Path("/proc/stat").read_text().split("\n", 1)[0].split()[1:])
        process = 0
        for task in Path(f"/proc/{pid}/task").iterdir():
            values = Path(task / "stat").read_text().split()
            process += int(values[13]) + int(values[14])
        return process, total
    except (OSError, ValueError, IndexError):
        return None


def docker_memory(name: str) -> dict[str, Any]:
    stats = run(["docker", "stats", "--no-stream", "--format", "{{json .}}", name], check=False)
    if stats.returncode or not stats.stdout.strip():
        return {}
    try:
        value = json.loads(stats.stdout.strip())
    except json.JSONDecodeError:
        return {"raw": stats.stdout.strip()}
    return {"raw": value, "memory_usage": value.get("MemUsage"), "memory_percent": value.get("MemPerc")}


def env_file() -> Path:
    path = OUTPUT / "benchmark.env"
    path.parent.mkdir(parents=True, exist_ok=True)
    source = (ROOT / ".env").read_text(encoding="utf-8") if (ROOT / ".env").is_file() else (ROOT / ".env.example").read_text(encoding="utf-8")
    overrides = {
        "MODEL_DIR": "/opt/voight/models", "RUNTIME_TARGET": "cpu", "OCR_DEVICE": "cpu", "CPU_THREADS": "4",
        "LOCALIZATION_BATCH_SIZE": "4", "TEXT_DETECTION_BATCH_SIZE": "1", "TEXT_RECOGNITION_BATCH_SIZE": "2",
        "MRZ_RECOGNITION_BATCH_SIZE": "2", "TEXT_RECOGNITION_PROCESSES": "1", "TEXT_RECOGNITION_PACKING": "fixed-width",
        "TEXT_DETECTOR_MODEL": "PP-OCRv6_medium_det", "TEXT_RECOGNIZER_MODEL": "latin_PP-OCRv5_mobile_rec",
        "DOCALIGNER_MODEL": "fastvit_sa24", "MRZ_RECOGNIZER_BACKEND": "generic-paddle",
    }
    lines = []
    seen = set()
    for line in source.splitlines():
        key = line.split("=", 1)[0] if "=" in line and not line.lstrip().startswith("#") else ""
        if key in overrides:
            lines.append(f"{key}={overrides[key]}")
            seen.add(key)
        else:
            lines.append(line)
    lines.extend(f"{key}={value}" for key, value in overrides.items() if key not in seen)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def container_probe(name: str) -> dict[str, Any]:
    code = f"""
import json, os, cv2, paddle, onnxruntime as ort
from app.config import Settings
from app.models import Models
s = Settings.from_env()
m = Models(s)
print(json.dumps({{
    "settings": {{
        "cpu_threads": s.runtime.cpu_threads,
        "localization_batch_size": s.runtime.localization_batch_size,
        "text_detection_batch_size": s.runtime.text_detection_batch_size,
        "text_recognition_batch_size": s.runtime.text_recognition_batch_size,
        "mrz_recognition_batch_size": s.runtime.mrz_recognition_batch_size,
        "text_recognition_processes": s.runtime.text_recognition_processes,
        "packing": s.runtime.text_recognition_packing,
    }},
    "models": {{
        "document_localizer": s.driving_license.aligner_model,
        "text_detector": s.models.text_detector.model,
        "text_recognizer": s.models.text_recognizer.model,
        "mrz_recognizer": s.models.mrz.recognizer_backend,
        "mrz_recognizer_model": s.models.text_recognizer.model,
    }},
    "thread_control": {{
        "paddle_text_recognition_cpu_threads": s.runtime.cpu_threads,
        "paddle_text_detection_cpu_threads": "not passed by app; OMP_NUM_THREADS governs native Paddle/OpenMP path",
        "onnxruntime_intra_op_num_threads": m._onnx_session_options().get("intra_op_num_threads"),
        "onnxruntime_inter_op_num_threads": ort.SessionOptions().inter_op_num_threads,
        "opencv_threads": cv2.getNumThreads(),
    }},
    "onnx_session_options": m._onnx_session_options(),
    "onnx_default_session_options": {{
        "intra_op_num_threads": ort.SessionOptions().intra_op_num_threads,
        "inter_op_num_threads": ort.SessionOptions().inter_op_num_threads,
    }},
    "paddle_device": paddle.get_device(),
    "paddle_set_num_threads_available": hasattr(__import__("paddle.base.core", fromlist=["core"]), "set_num_threads"),
    "opencv": {{
        "getNumThreads": cv2.getNumThreads(),
        "useOptimized": cv2.useOptimized(),
        "parallel_framework": next((x.split(":", 1)[1].strip() for x in cv2.getBuildInformation().splitlines() if x.strip().startswith("Parallel framework:")), None),
    }},
    "environment": {{k: os.environ.get(k) for k in {repr(ENV_KEYS)}}},
}}))
"""
    result = run(["docker", "exec", name, "python", "-c", code], check=False, timeout=120)
    if result.returncode:
        return {"error": result.stderr[-2000:]}
    lines = [line for line in result.stdout.splitlines() if line.startswith("{")]
    return json.loads(lines[-1]) if lines else {"error": result.stdout[-2000:]}


def dataset_records(kind: str) -> list[dict[str, Any]]:
    folder = {"passport": "passport", "id-card": "id_card", "driving-license": "driving_license"}[kind]
    annotation_dir = DATASET / "annotations" / folder
    records = []
    for annotation_path in sorted(annotation_dir.glob("*.json")):
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        if kind == "id-card":
            images = {role: DATASET / value for role, value in annotation["images"].items()}
        else:
            images = {"image": DATASET / annotation["images"]["image"]}
        if any(not path.is_file() or sha256(path) != annotation["image_sha256"][role] for role, path in images.items()):
            raise RuntimeError(f"dataset/annotation hash mismatch: {annotation_path}")
        records.append({"id": annotation["id"], "images": images, "truth": annotation})
    expected = EXPECTED_DOCUMENTS[kind]
    if len(records) != expected or sum(len(record["images"]) for record in records) != EXPECTED_IMAGES[kind]:
        raise RuntimeError(f"{kind} dataset count mismatch: documents={len(records)} images={sum(len(record['images']) for record in records)}")
    return records


def dataset_manifest() -> dict[str, Any]:
    result = {}
    for kind in DOCS:
        records = dataset_records(kind)
        result[kind] = {
            "logical_documents": len(records),
            "physical_images": sum(len(record["images"]) for record in records),
            "documents": [{"id": record["id"], "images": {role: str(path.relative_to(DATASET)) for role, path in record["images"].items()}, "sha256": {role: sha256(path) for role, path in record["images"].items()}} for record in records],
        }
    return result


def archive_id() -> tuple[bytes, list[dict[str, Any]]]:
    output = BytesIO()
    records = dataset_records("id-card")
    with zipfile.ZipFile(output, "w") as archive:
        for record in records:
            for role, path in record["images"].items():
                archive.writestr(f"{record['id']}/{role}{path.suffix}", path.read_bytes())
    return output.getvalue(), [{"id": record["id"], "images": list(record["images"])} for record in records]


def request_workload(kind: str, label: str) -> tuple[Any, dict[str, Any]]:
    records = dataset_records(kind)
    manifest = [{"id": record["id"], "images": {role: path.name for role, path in record["images"].items()}, "sha256": {role: sha256(path) for role, path in record["images"].items()}} for record in records]
    info = {"verified": True, "expected_logical_documents": EXPECTED_DOCUMENTS[kind], "expected_physical_images": EXPECTED_IMAGES[kind], "submitted_logical_documents": len(records), "submitted_physical_images": sum(len(record["images"]) for record in records), "documents": manifest}
    if kind == "id-card":
        archive, archive_manifest = archive_id()
        info["archive_documents"] = archive_manifest
        return {"archive": ("cards.zip", archive, "application/zip")}, info
    files = []
    for record in records:
        path = record["images"]["image"]
        mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        files.append(("images", (f"{label}-{record['id']}{path.suffix}", path.read_bytes(), mime)))
    return files, info


def request(name: str, kind: str, *, label: str, timeout: float = TIMEOUT) -> tuple[dict[str, Any], dict[str, Any]]:
    files, request_info = request_workload(kind, label)
    if request_info["submitted_logical_documents"] != request_info["expected_logical_documents"] or request_info["submitted_physical_images"] != request_info["expected_physical_images"]:
        raise RuntimeError(f"request workload count mismatch: {request_info}")
    try:
        response = requests.post(f"http://127.0.0.1:{PORT}/v1/ocr/{kind}/batch", files=files, timeout=timeout)
    except requests.RequestException as error:
        payload = {"error": f"{type(error).__name__}: {error}"}
        return {"status_code": None, "ok": False, "payload": payload, "request_verification": request_info}, payload
    try:
        payload = response.json()
    except ValueError:
        payload = {"error": response.text[:2000]}
    request_info["response_item_count"] = len(payload.get("items", [])) if isinstance(payload, dict) else None
    request_info["response_total"] = payload.get("total") if isinstance(payload, dict) else None
    request_info["response_counts_match"] = request_info["response_item_count"] == request_info["expected_logical_documents"] and request_info["response_total"] == request_info["expected_logical_documents"]
    return {"status_code": response.status_code, "ok": response.ok, "payload": payload, "request_verification": request_info}, payload


def batches(stage: Any) -> Any:
    if not isinstance(stage, dict):
        return {}
    if "tensor_batch_sizes" in stage:
        return stage.get("tensor_batch_sizes", [])
    return {name: value.get("tensor_batch_sizes", []) for name, value in stage.items() if isinstance(value, dict)}


def stage_times(payload: dict[str, Any]) -> dict[str, float]:
    diagnostics = payload.get("diagnostics", {})
    pipeline = diagnostics.get("pipeline", {})
    localization = sum(float(value.get("wall_seconds", 0.0)) for value in diagnostics.get("localization", {}).values() if isinstance(value, dict))
    detection = float(diagnostics.get("text_detection", {}).get("wall_seconds", 0.0))
    recognition = float(diagnostics.get("text_recognition", {}).get("wall_seconds", 0.0))
    return {"e2e_seconds": float(payload.get("total_seconds", 0.0)), "localization_seconds": localization, "detection_seconds": detection, "recognition_seconds": recognition}


def output_only(item: dict[str, Any]) -> dict[str, Any]:
    result = item.get("result")
    if isinstance(result, dict):
        result = {key: value for key, value in result.items() if key != "timings"}
    return {"success": item.get("success"), "result": result, "error": item.get("error")}


def semantic_output(value: Any, parent: str = "") -> Any:
    if isinstance(value, dict):
        return {key: semantic_output(item, key) for key, item in value.items() if not (parent == "confidence" and key == "score")}
    if isinstance(value, list):
        return [semantic_output(item, parent) for item in value]
    return value


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def levenshtein(left: str, right: str) -> int:
    previous = list(range(len(right) + 1))
    for index, left_char in enumerate(left, 1):
        current = [index]
        for right_index, right_char in enumerate(right, 1):
            current.append(min(current[-1] + 1, previous[right_index] + 1, previous[right_index - 1] + (left_char != right_char)))
        previous = current
    return previous[-1]


def correctness(kind: str, payload: dict[str, Any]) -> dict[str, Any]:
    truths = [record["truth"] for record in dataset_records(kind)]
    results = []
    fields_exact = fields_evaluated = chars_correct = chars_evaluated = mrz_exact_items = mrz_evaluated_items = mrz_chars_correct = mrz_chars_evaluated = 0
    for index, item in enumerate(payload.get("items", [])):
        truth = truths[index] if index < len(truths) else {}
        fields = truth.get("fields", {})
        actual = {name: value.get("value") if isinstance(value, dict) else value for name, value in item.get("result", {}).get("fields", {}).items()} if item.get("success") else {}
        expected_fields = {name: value.get("value") for name, value in fields.items() if value.get("state") == "value"}
        exact = sum(actual.get(name) == value for name, value in expected_fields.items())
        evaluated = len(expected_fields)
        field_chars_correct = field_chars_total = 0
        for name, expected in expected_fields.items():
            actual_value = str(actual.get(name) or "")
            expected_value = str(expected)
            field_chars_correct += max(len(expected_value), len(actual_value)) - levenshtein(expected_value, actual_value)
            field_chars_total += max(len(expected_value), len(actual_value))
        result = item.get("result") or {}
        mrz = (result.get("mrz") or {}).get("raw_lines", []) if item.get("success") else []
        expected_mrz = truth.get("mrz", {}).get("lines", [])
        mrz_evaluated = bool(expected_mrz)
        mrz_expected_text = "\n".join(expected_mrz)
        mrz_actual_text = "\n".join(mrz)
        mrz_chars_total = max(len(mrz_expected_text), len(mrz_actual_text)) if mrz_evaluated else 0
        mrz_chars_good = mrz_chars_total - levenshtein(mrz_expected_text, mrz_actual_text) if mrz_evaluated else 0
        row = {"document_id": truth.get("id"), "success": bool(item.get("success")), "fields_exact": exact, "fields_evaluated": evaluated, "field_exact_rate": exact / evaluated if evaluated else None, "field_characters_correct": field_chars_correct, "field_characters_evaluated": field_chars_total, "character_accuracy": field_chars_correct / field_chars_total if field_chars_total else None, "mrz_exact": mrz == expected_mrz if mrz_evaluated else None, "mrz_evaluated": mrz_evaluated, "mrz_expected_lines": len(expected_mrz), "mrz_actual_lines": len(mrz), "mrz_characters_correct": mrz_chars_good, "mrz_characters_evaluated": mrz_chars_total, "mrz_character_accuracy": mrz_chars_good / mrz_chars_total if mrz_chars_total else None}
        results.append(row)
        fields_exact += exact
        fields_evaluated += evaluated
        chars_correct += field_chars_correct
        chars_evaluated += field_chars_total
        mrz_exact_items += row["mrz_exact"] is True
        mrz_evaluated_items += mrz_evaluated
        mrz_chars_correct += mrz_chars_good
        mrz_chars_evaluated += mrz_chars_total
    return {"truth_annotations": [truth.get("id") for truth in truths], "items": results, "fields_exact": fields_exact, "fields_evaluated": fields_evaluated, "field_exact_rate": fields_exact / fields_evaluated if fields_evaluated else None, "characters_correct": chars_correct, "characters_evaluated": chars_evaluated, "character_accuracy": chars_correct / chars_evaluated if chars_evaluated else None, "mrz_exact_items": mrz_exact_items, "mrz_evaluated_items": mrz_evaluated_items, "mrz_characters_correct": mrz_chars_correct, "mrz_characters_evaluated": mrz_chars_evaluated, "mrz_character_accuracy": mrz_chars_correct / mrz_chars_evaluated if mrz_chars_evaluated else None}


def measure_request(name: str, kind: str, label: str, repeat: int) -> tuple[dict[str, Any], dict[str, Any]]:
    pid = container_pid(name)
    before = proc_cpu(pid) if pid else None
    started = time.perf_counter()
    request_meta, payload = request(name, kind, label=label)
    elapsed = time.perf_counter() - started
    after = proc_cpu(pid) if pid else None
    cpu_util = None
    if before and after and elapsed > 0:
        process_ticks = after[0] - before[0]
        host_ticks = after[1] - before[1]
        cpu_util = process_ticks / host_ticks * 100 * (os.cpu_count() or 1) if host_ticks else None
    diagnostics = payload.get("diagnostics", {}) if isinstance(payload, dict) else {}
    item_outputs = [output_only(item) for item in payload.get("items", [])] if isinstance(payload, dict) else []
    request_info = request_meta["request_verification"]
    line_filter = diagnostics.get("line_filter") or {}
    recognition_calls = (diagnostics.get("text_recognition") or {}).get("calls", [])
    mrz_tensor_batches = [call.get("tensor_batch_size") for call in recognition_calls if call.get("role") == "mrz"]
    correctness_data = correctness(kind, payload) if request_meta["ok"] else {}
    row = {
        "thread_count": int(label.split("-", 1)[0]), "document_type": kind, "repeat": repeat,
        "request_seconds": elapsed, "status_code": request_meta["status_code"], "status": "ok" if request_meta["ok"] else "failed",
        "succeeded": payload.get("succeeded"), "failed": payload.get("failed"), "server_total_seconds": payload.get("total_seconds"),
        **stage_times(payload), "docs_per_second": EXPECTED_DOCUMENTS[kind] / float(payload.get("total_seconds", elapsed) or elapsed),
        "cpu_utilization_percent": cpu_util, "peak_rss_mb": diagnostics.get("process_peak_rss_mb"),
        "cpu_utilization_scope": "container main process over request; host jiffies",
        "request_verification": request_info,
        "expected_logical_documents": EXPECTED_DOCUMENTS[kind], "expected_physical_images": EXPECTED_IMAGES[kind],
        "actual_logical_documents": payload.get("total"), "actual_response_items": len(payload.get("items", [])),
        "actual_physical_images": request_info.get("submitted_physical_images"), "actual_detected_line_count": line_filter.get("detected_line_count"),
        "actual_recognition_crop_count": line_filter.get("recognition_candidate_count"), "actual_filtered_crop_count": line_filter.get("filtered_before_recognition_count"),
        "sample_counts": diagnostics.get("sample_counts"), "line_counts_by_role": diagnostics.get("line_counts_by_role"),
        "localization_tensor_batches": batches(diagnostics.get("localization")),
        "detection_tensor_batches": batches(diagnostics.get("text_detection")),
        "recognition_tensor_batches": batches(diagnostics.get("text_recognition")),
        "mrz_recognition_tensor_batches": batches(diagnostics.get("mrz_recognition")),
        "actual_mrz_tensor_batches": mrz_tensor_batches,
        "actual_tensor_batches": {"localization": batches(diagnostics.get("localization")), "detection": batches(diagnostics.get("text_detection")), "recognition": batches(diagnostics.get("text_recognition")), "mrz_recognition": batches(diagnostics.get("mrz_recognition"))},
        "correctness": correctness_data,
        "output_digests": [digest(item) for item in item_outputs],
        "semantic_output_digests": [digest(semantic_output(item)) for item in item_outputs],
        "output": item_outputs,
        "failure": payload.get("error") if isinstance(payload, dict) else str(payload),
    }
    return row, payload


def lifecycle(name: str, run_dir: Path) -> dict[str, Any]:
    memory_before_stop = docker_memory(name)
    stop = run(["docker", "stop", "-t", "10", name], check=False, timeout=30)
    wait = run(["docker", "wait", name], check=False, timeout=30)
    memory_after_stop = docker_memory(name)
    pid_after_stop = container_pid(name)
    pid_before_rm = pid_after_stop
    logs = run(["docker", "logs", name], check=False, timeout=30)
    (run_dir / "server.log").write_text(logs.stdout + logs.stderr, encoding="utf-8")
    rm = run(["docker", "rm", "-f", name], check=False, timeout=30)
    after = run(["docker", "inspect", name], check=False, timeout=30)
    return {"stop_returncode": stop.returncode, "wait_output": wait.stdout.strip(), "pid_after_stop": pid_after_stop, "memory_before_stop": memory_before_stop, "memory_after_stop": memory_after_stop, "inspect_after_remove_returncode": after.returncode, "container_removed": after.returncode != 0, "process_termination_verified": stop.returncode == 0 and pid_after_stop == 0 and after.returncode != 0, "ram_release_verification": "container removed; host MemAvailable is recorded but OS cache reclamation is not asserted", "rm_returncode": rm.returncode, "host_mem_available_kib_after": available_memory_kib()}


def start(name: str, threads: int, env: Path) -> tuple[float, dict[str, Any], int]:
    command = ["docker", "run", "--detach", "--name", name, "--publish", f"{PORT}:8000", "--env-file", str(env), "--env", f"CPU_THREADS={threads}", IMAGE]
    started = time.perf_counter()
    result = run(command, timeout=120)
    container = result.stdout.strip()
    deadline = time.monotonic() + 900
    ready = None
    while time.monotonic() < deadline:
        try:
            response = requests.get(f"http://127.0.0.1:{PORT}/v1/health/ready", timeout=30)
            if response.ok:
                ready = response.json()
                break
        except requests.RequestException:
            pass
        time.sleep(2)
    if ready is None:
        logs = run(["docker", "logs", name], check=False).stdout[-4000:]
        raise RuntimeError(f"server did not become ready: {logs}")
    return time.perf_counter() - started, ready, container_pid(name) or 0


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def compact_comparison(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    baseline = {kind: [row["output_digests"] for row in rows if row["thread_count"] == 4 and row["document_type"] == kind] for kind in DOCS}
    semantic_baseline = {kind: [row["semantic_output_digests"] for row in rows if row["thread_count"] == 4 and row["document_type"] == kind] for kind in DOCS}
    result = []
    for thread_count in THREADS:
        for kind in DOCS:
            selected = [row for row in rows if row["thread_count"] == thread_count and row["document_type"] == kind]
            if not selected:
                continue
            good = [row for row in selected if row["status"] == "ok"]
            correctness_rows = [row["correctness"] for row in good if row.get("correctness")]
            base = baseline[kind]
            semantic_base = semantic_baseline[kind]
            differences = sum(row["output_digests"] not in base for row in selected) if base else None
            semantic_differences = sum(row["semantic_output_digests"] not in semantic_base for row in selected) if semantic_base else None
            result.append({
                "thread_count": thread_count, "document_type": kind, "measured_repeats": len(selected),
                "successful_repeats": len(good), "failed_repeats": len(selected) - len(good),
                "e2e_seconds_median": statistics.median(row["e2e_seconds"] for row in good) if good else None,
                "docs_per_second_median": statistics.median(row["docs_per_second"] for row in good) if good else None,
                "localization_seconds_median": statistics.median(row["localization_seconds"] for row in good) if good else None,
                "detection_seconds_median": statistics.median(row["detection_seconds"] for row in good) if good else None,
                "recognition_seconds_median": statistics.median(row["recognition_seconds"] for row in good) if good else None,
                "peak_rss_mb_max": max((row["peak_rss_mb"] or 0.0 for row in good), default=None),
                "cpu_utilization_percent_median": statistics.median(row["cpu_utilization_percent"] for row in good if row["cpu_utilization_percent"] is not None) if any(row["cpu_utilization_percent"] is not None for row in good) else None,
                "expected_logical_documents": selected[0].get("expected_logical_documents"), "expected_physical_images": selected[0].get("expected_physical_images"),
                "actual_logical_documents": sorted({row.get("actual_logical_documents") for row in good}), "actual_physical_images": sorted({row.get("actual_physical_images") for row in good}),
                "actual_detected_line_count_median": statistics.median(row["actual_detected_line_count"] for row in good if row.get("actual_detected_line_count") is not None) if any(row.get("actual_detected_line_count") is not None for row in good) else None,
                "actual_recognition_crop_count_median": statistics.median(row["actual_recognition_crop_count"] for row in good if row.get("actual_recognition_crop_count") is not None) if any(row.get("actual_recognition_crop_count") is not None for row in good) else None,
                "fields_exact": statistics.median(row.get("fields_exact", 0) for row in correctness_rows) if correctness_rows else None,
                "fields_evaluated": statistics.median(row.get("fields_evaluated", 0) for row in correctness_rows) if correctness_rows else None,
                "field_exact_rate": statistics.median(row.get("field_exact_rate", 0.0) for row in correctness_rows) if correctness_rows else None,
                "characters_correct": statistics.median(row.get("characters_correct", 0) for row in correctness_rows) if correctness_rows else None,
                "characters_evaluated": statistics.median(row.get("characters_evaluated", 0) for row in correctness_rows) if correctness_rows else None,
                "character_accuracy": statistics.median(row.get("character_accuracy", 0.0) for row in correctness_rows) if correctness_rows else None,
                "mrz_exact_items": statistics.median(row.get("mrz_exact_items", 0) for row in correctness_rows) if correctness_rows else None,
                "mrz_evaluated_items": statistics.median(row.get("mrz_evaluated_items", 0) for row in correctness_rows) if correctness_rows else None,
                "mrz_character_accuracy": statistics.mean(row["mrz_character_accuracy"] for row in correctness_rows if row.get("mrz_character_accuracy") is not None) if any(row.get("mrz_character_accuracy") is not None for row in correctness_rows) else None,
                "baseline_4_threads": bool(base), "output_differences_vs_4_threads": differences,
                "semantic_output_differences_vs_4_threads": semantic_differences,
                "confidence_only_differences_vs_4_threads": differences - semantic_differences if differences is not None and semantic_differences is not None else None,
                "actual_tensor_batches": good[0].get("actual_tensor_batches") if good else None,
                "actual_mrz_tensor_batches": good[0].get("actual_mrz_tensor_batches") if good else None,
            })
    return result


def main() -> int:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    env = env_file()
    system = {"schema_version": 2, "benchmark_started_utc": datetime.now(timezone.utc).isoformat(), "cpu": host_cpus(), "image": IMAGE, "image_inspect": run(["docker", "image", "inspect", IMAGE, "--format", "{{json .}}"], check=False).stdout.strip(), "dataset": {"root": str(DATASET), "expected_logical_documents": {kind: EXPECTED_DOCUMENTS[kind] for kind in DOCS}, "expected_physical_images": {kind: EXPECTED_IMAGES[kind] for kind in DOCS}, "manifest": dataset_manifest()}, "fixed_configuration": {"models": {"document_localizer": "fastvit_sa24", "text_detector": "PP-OCRv6_medium_det", "text_recognizer": "latin_PP-OCRv5_mobile_rec", "mrz_recognizer": "generic-paddle/Latin"}, "localization_batch": 4, "detection_batch": 1, "recognition_batch": 2, "mrz_recognition_batch": 2, "recognition_processes": 1, "packing": "fixed-width"}, "host_environment_thread_variables": {key: os.environ.get(key) for key in ENV_KEYS}}
    (OUTPUT / "system.json").write_text(json.dumps(system, indent=2), encoding="utf-8")
    all_rows: list[dict[str, Any]] = []
    for threads in THREADS:
        name = f"voight-cpu-thread-{threads}"
        run_dir = OUTPUT / "runs" / str(threads)
        run_dir.mkdir(parents=True, exist_ok=True)
        for old_artifact in run_dir.iterdir():
            if old_artifact.is_file():
                old_artifact.unlink()
        before_mem = available_memory_kib()
        started = ready = pid = None
        lifecycle_data: dict[str, Any] = {}
        try:
            startup_seconds, ready, pid = start(name, threads, env)
            probe = container_probe(name)
            (run_dir / "startup.json").write_text(json.dumps({"startup_seconds": startup_seconds, "pid": pid, "ready": ready, "effective": probe}, indent=2), encoding="utf-8")
            for kind in DOCS:
                meta, payload = request(name, kind, label=f"{threads}-warmup")
                (run_dir / f"warmup_{kind}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
                if not meta["ok"]:
                    raise RuntimeError(f"warm-up failed for {kind}: {meta}")
            for kind in DOCS:
                for repeat in range(1, REPEATS + 1):
                    row, payload = measure_request(name, kind, f"{threads}-{kind}", repeat)
                    all_rows.append(row)
                    (run_dir / f"{kind}_repeat_{repeat}.json").write_text(json.dumps({"measurement": row, "response": payload}, indent=2), encoding="utf-8")
                    print(f"threads={threads} {kind} repeat={repeat} total={row['server_total_seconds']} status={row['status']}", flush=True)
        except Exception as error:
            (run_dir / "run_error.txt").write_text(f"{type(error).__name__}: {error}\n", encoding="utf-8")
            print(f"threads={threads} failed: {error}", file=sys.stderr, flush=True)
        finally:
            lifecycle_data = lifecycle(name, run_dir) if container_pid(name) is not None else {"container_removed": True, "process_termination_verified": False, "note": "container was absent before cleanup"}
            lifecycle_data["host_mem_available_kib_before"] = before_mem
            (run_dir / "lifecycle.json").write_text(json.dumps(lifecycle_data, indent=2), encoding="utf-8")
            (run_dir / "run.json").write_text(json.dumps({"thread_count": threads, "startup_pid": pid, "startup_ready": ready is not None, "warmup_count": len(list(run_dir.glob("warmup_*.json"))), "measured_repeat_count": len(list(run_dir.glob("*_repeat_*.json"))), "lifecycle": lifecycle_data}, indent=2), encoding="utf-8")
    write_csv(OUTPUT / "raw_results.csv", all_rows)
    write_csv(OUTPUT / "comparison.csv", compact_comparison(all_rows))
    (OUTPUT / "raw_results.json").write_text(json.dumps(all_rows, indent=2), encoding="utf-8")
    reference_path = ROOT / "outputs/benchmarks/09.batch-size-sweep/20260822T140836Z/summary.json"
    reference = json.loads(reference_path.read_text(encoding="utf-8")) if reference_path.is_file() else []
    reference = {row["document_type"] if "document_type" in row else {"id_card": "id-card", "driving_license": "driving-license"}.get(row.get("document_type"), row.get("document_type")): row for row in reference if row.get("configuration") == "localization_4_detection_1_recognition_2"}
    current = {row["document_type"]: row for row in compact_comparison(all_rows) if row["thread_count"] == 4}
    sanity_comparisons = {kind: {"current_e2e_seconds_median": current[kind]["e2e_seconds_median"], "reference_e2e_seconds_median": reference.get(kind, {}).get("median_total_latency_seconds"), "relative_delta": (current[kind]["e2e_seconds_median"] - reference[kind]["median_total_latency_seconds"]) / reference[kind]["median_total_latency_seconds"] if kind in reference else None} for kind in DOCS}
    sanity = {"reference_artifact": str(reference_path), "reference_configuration": "localization_4_detection_1_recognition_2", "comparisons": sanity_comparisons, "assessment": "consistent_with_reference" if reference and max(abs(item["relative_delta"]) for item in sanity_comparisons.values() if item["relative_delta"] is not None) < 0.2 else "investigate_difference"}
    (OUTPUT / "sanity_check.json").write_text(json.dumps(sanity, indent=2), encoding="utf-8")
    (OUTPUT / "README.md").write_text("""# CPU thread benchmark\n\nCommand: `uv run --no-sync python benchmarks/historical/cpu_thread_benchmark.py`\n\nEach thread count uses a fresh `voight:cpu` container, one warm-up request for each document type, three measured requests for each type, and complete stop/remove verification before the next count. Every request is verified against the complete dataset: 9 passports/9 images, 4 logical ID cards/8 images, or 7 driving licences/7 images. ID cards are submitted as one archive containing four directories with front/back sides.\n\nOnly `CPU_THREADS` varies between runs. The benchmark environment fixes the finalized models, CPU target, localization batch 4, detection batch 1, text and MRZ recognition batch 2, fixed-width packing, one recognition process, and all other listed model/runtime settings.\n\n`system.json` records the machine, dataset manifest, and effective runtime probe. `runs/<threads>/` contains startup, warm-up verification, raw measured responses, server logs, and lifecycle evidence. `raw_results.json`/`raw_results.csv` contain per-repeat measurements; `comparison.csv` is the compact comparison table. Ground-truth correctness is separate from output equality versus the 4-thread baseline.\n""", encoding="utf-8")
    print(f"artifacts: {OUTPUT}")
    return 0 if len(all_rows) == len(THREADS) * len(DOCS) * REPEATS else 2


if __name__ == "__main__":
    raise SystemExit(main())
