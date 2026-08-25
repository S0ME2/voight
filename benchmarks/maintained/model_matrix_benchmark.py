"""Run the prepared CPU model matrix with one fresh Voight server per row."""

from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import socket
import statistics
import subprocess
import sys
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.maintained.pipeline_breakdown import annotation_truth, discover_dataset, validate_and_manifest, _distance

PADDLE_DETECTORS = (
    "PP-OCRv6_medium_det", "PP-OCRv6_small_det", "PP-OCRv6_tiny_det",
    "PP-OCRv5_server_det", "PP-OCRv5_mobile_det", "PP-OCRv4_server_det", "PP-OCRv4_mobile_det",
)
PADDLE_RECOGNIZERS = (
    "PP-OCRv6_medium_rec", "PP-OCRv6_small_rec", "PP-OCRv6_tiny_rec", "PP-OCRv5_server_rec",
    "PP-OCRv5_mobile_rec", "latin_PP-OCRv5_mobile_rec", "en_PP-OCRv5_mobile_rec",
    "cyrillic_PP-OCRv5_mobile_rec", "eslav_PP-OCRv5_mobile_rec", "PP-OCRv4_server_rec",
    "PP-OCRv4_mobile_rec", "en_PP-OCRv4_mobile_rec",
)


@dataclass(frozen=True)
class Candidate:
    category: str
    model: str
    env: dict[str, str]
    blocked: str | None = None


def candidates() -> list[Candidate]:
    base = {"TEXT_DETECTOR_MODEL": "PP-OCRv6_medium_det", "TEXT_RECOGNIZER_MODEL": "PP-OCRv6_medium_rec", "DOCALIGNER_MODEL": "fastvit_sa24", "DOCALIGNER_MODEL_TYPE": "heatmap", "MRZ_RECOGNIZER_BACKEND": "generic-paddle", "MRZ_RECOGNIZER_MODEL": "20250221"}
    rows = [Candidate("baseline", "PP-OCRv6_medium_det + PP-OCRv6_medium_rec", base)]
    rows += [Candidate("detector", name, {**base, "TEXT_DETECTOR_MODEL": name}) for name in PADDLE_DETECTORS]
    rows += [Candidate("recognizer", name, {**base, "TEXT_RECOGNIZER_MODEL": name}) for name in PADDLE_RECOGNIZERS]
    for name in ("fastvit_sa24", "mobilenetv2_140", "fastvit_t8", "lcnet100", "lcnet050"):
        blocked = "not present in pinned docaligner heatmap config" if name in {"mobilenetv2_140", "lcnet050"} else None
        rows.append(Candidate("docaligner_heatmap", name, {**base, "DOCALIGNER_MODEL": name}, blocked))
    rows.append(Candidate("docaligner_point", "lcnet050", {**base, "DOCALIGNER_MODEL": "lcnet050", "DOCALIGNER_MODEL_TYPE": "point"}))
    rows += [
        Candidate("mrz", "detection_20250222 + generic-paddle", base),
        Candidate("mrz", "detection_20250222 + recognition_20250221", {**base, "MRZ_RECOGNIZER_BACKEND": "mrzscanner"}),
        Candidate("mrz", "detection_20250222 + spotting_20240919", {**base, "MRZ_RECOGNIZER_BACKEND": "mrzscanner-spotting", "MRZ_RECOGNIZER_MODEL": "20240919"}),
    ]
    return rows


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", ".paddlex")))
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/benchmarks/model_matrix")
    parser.add_argument("--port", type=int, default=8011)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--recognizers", nargs="+", help="run only these recognizer model names")
    args = parser.parse_args()
    if args.repeats < 3 or args.warmup < 1:
        parser.error("use at least one warm-up and three measured repeats")
    return args


def _rss(pid: int) -> int | None:
    try:
        for line in (Path(f"/proc/{pid}/status")).read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    return None


def _available_memory() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except (FileNotFoundError, ValueError):
        return None
    return None


def _pids_in_group(pgid: int) -> list[int]:
    found = []
    for path in Path("/proc").glob("[0-9]*"):
        try:
            rest = path.joinpath("stat").read_text().split(") ", 1)[1].split()
            if int(rest[2]) == pgid:
                found.append(int(path.name))
        except (FileNotFoundError, PermissionError, ValueError, IndexError):
            continue
    return found


def _port_open(port: int) -> bool:
    with socket.socket() as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


class Server:
    def __init__(self, args: argparse.Namespace, candidate_dir: Path, env: dict[str, str]):
        self.args, self.candidate_dir, self.env = args, candidate_dir, env
        self.process: subprocess.Popen[str] | None = None
        self.log = candidate_dir / "server.log"
        self.peak_rss = 0

    def start(self) -> dict[str, Any]:
        if _port_open(self.args.port):
            raise RuntimeError(f"benchmark port {self.args.port} is already listening")
        server_env = os.environ.copy()
        server_env.update({
            "RUNTIME_TARGET": "cpu", "OCR_DEVICE": "cpu", "MODEL_DIR": str(self.args.model_dir),
            "PRELOAD": "true", "LOGGING": "false", "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
            "CPU_THREADS": os.getenv("CPU_THREADS", "4"), "LOCALIZATION_BATCH_SIZE": os.getenv("LOCALIZATION_BATCH_SIZE", "16"),
            "TEXT_DETECTION_BATCH_SIZE": os.getenv("TEXT_DETECTION_BATCH_SIZE", "16"), "TEXT_RECOGNITION_BATCH_SIZE": os.getenv("TEXT_RECOGNITION_BATCH_SIZE", "32"),
            "MRZ_RECOGNITION_BATCH_SIZE": os.getenv("MRZ_RECOGNITION_BATCH_SIZE", "16"), "TEXT_RECOGNITION_PROCESSES": "1",
        })
        server_env.update(self.env)
        self.log.parent.mkdir(parents=True, exist_ok=True)
        handle = self.log.open("w", encoding="utf-8")
        self.process = subprocess.Popen(
            [sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", str(self.args.port), "--workers", "1"],
            cwd=ROOT, env=server_env, stdout=handle, stderr=subprocess.STDOUT,
            start_new_session=True, text=True,
        )
        handle.close()
        deadline = time.monotonic() + self.args.timeout
        ready = None
        error = None
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"server exited with code {self.process.returncode}; see {self.log}")
            self._sample()
            try:
                response = requests.get(f"http://127.0.0.1:{self.args.port}/v1/health/ready", timeout=5)
                if response.ok:
                    ready = response.json()
                    break
                error = response.text[:1000]
            except requests.RequestException as exc:
                error = str(exc)
            time.sleep(1)
        if ready is None:
            raise RuntimeError(f"server readiness timed out: {error}; see {self.log}")
        self._verify_loaded(ready)
        return ready

    def _verify_loaded(self, ready: dict[str, Any]) -> None:
        models = ready.get("models", {})
        expected = {
            "text_detector": (("model", self.env["TEXT_DETECTOR_MODEL"]),),
            "text_recognizer": (("model", self.env["TEXT_RECOGNIZER_MODEL"]),),
            "document_localizer": (("model_cfg", self.env["DOCALIGNER_MODEL"]), ("model_type", self.env["DOCALIGNER_MODEL_TYPE"])),
            "mrz_localizer": (("model_cfg", "20250222"),),
            "mrz_recognizer": (("backend", self.env["MRZ_RECOGNIZER_BACKEND"]),),
        }
        for section, checks in expected.items():
            for key, value in checks:
                if models.get(section, {}).get(key) != value:
                    raise RuntimeError(f"loaded configuration mismatch for {section}.{key}: expected {value!r}, got {models.get(section)}")
        for section, data in models.items():
            if not data.get("loaded"):
                raise RuntimeError(f"model section did not report loaded: {section}: {data}")
        det_path = models["text_detector"].get("path")
        rec_path = models["text_recognizer"].get("path")
        if not det_path or Path(det_path).name != self.env["TEXT_DETECTOR_MODEL"]:
            raise RuntimeError(f"loaded detector path does not match candidate: {det_path}")
        if not rec_path or Path(rec_path).name != self.env["TEXT_RECOGNIZER_MODEL"]:
            raise RuntimeError(f"loaded recognizer path does not match candidate: {rec_path}")

    def _sample(self) -> None:
        if self.process is not None and (value := _rss(self.process.pid)) is not None:
            self.peak_rss = max(self.peak_rss, value)

    def stop(self) -> dict[str, Any]:
        if self.process is None:
            return {"server_pid": None, "cleanup_verified": True, "peak_process_memory_mb": None}
        pid = self.process.pid
        try:
            pgid = os.getpgid(pid)
        except ProcessLookupError:
            return {
                "server_pid": pid, "server_pgid": None, "server_returncode": self.process.returncode,
                "port_closed": not _port_open(self.args.port), "remaining_server_pids": [],
                "cleanup_verified": not _port_open(self.args.port),
                "peak_process_memory_mb": round(self.peak_rss / 1024 / 1024, 3),
            }
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
            "server_pid": pid, "server_pgid": pgid, "server_returncode": self.process.returncode,
            "port_closed": not _port_open(self.args.port), "remaining_server_pids": remaining,
            "cleanup_verified": not remaining and not _port_open(self.args.port),
            "peak_process_memory_mb": round(self.peak_rss / 1024 / 1024, 3),
        }


def _archive(documents: list[Any]) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for index, document in enumerate(documents):
            for role, path in document.paths:
                archive.writestr(f"card-{index:03d}/{role}{path.suffix.lower()}", path.read_bytes())
    return output.getvalue()


def _post(kind: str, documents: list[Any], port: int, timeout: float) -> tuple[dict[str, Any], float]:
    url = f"http://127.0.0.1:{port}/v1/ocr/{kind.replace('_', '-')}/batch"
    if kind == "id_card":
        files: Any = {"archive": ("cards.zip", _archive(documents), "application/zip")}
    else:
        files = [("images", (f"{doc.document_id}{doc.paths[0][1].suffix.lower()}", doc.paths[0][1].read_bytes(), "image/png" if doc.paths[0][1].suffix.lower() == ".png" else "image/jpeg")) for doc in documents]
    started = time.perf_counter()
    response = requests.post(url, files=files, timeout=timeout)
    elapsed = time.perf_counter() - started
    if not response.ok:
        raise RuntimeError(f"{kind} request failed {response.status_code}: {response.text[:1000]}")
    return response.json(), elapsed


def _score(kind: str, documents: list[Any], payload: dict[str, Any]) -> dict[str, Any]:
    fields = exact = chars = char_total = documents_exact = 0
    mrz_docs = mrz_found = mrz_full = mrz_lines = mrz_line_exact = mrz_chars = mrz_char_total = 0
    for document, item in zip(documents, payload.get("items", [])):
        truth = annotation_truth(document)
        result = item.get("result") or {}
        actual_fields = {name: entry.get("value") for name, entry in (result.get("fields") or {}).items()}
        doc_exact = bool(item.get("success", False))
        for name, entry in truth.get("fields", {}).items():
            if not isinstance(entry, dict) or entry.get("state") not in {"value", "empty"}:
                continue
            expected = entry.get("value") if entry.get("state") == "value" else None
            actual = actual_fields.get(name)
            fields += 1
            expected_text, actual_text = "" if expected is None else str(expected), "" if actual is None else str(actual)
            chars += max(len(expected_text), 1) - _distance(expected_text, actual_text)
            char_total += max(len(expected_text), 1)
            matched = actual == expected or (expected is None and actual in (None, ""))
            exact += matched
            doc_exact &= matched
        expected_lines = [line for line in truth.get("mrz", {}).get("lines", []) if isinstance(line, str)]
        if expected_lines:
            actual_lines = (result.get("mrz") or {}).get("raw_lines", [])
            mrz_docs += 1
            mrz_found += bool(actual_lines)
            full = actual_lines == expected_lines
            mrz_full += full
            doc_exact &= full
            for index, expected in enumerate(expected_lines):
                actual = actual_lines[index] if index < len(actual_lines) else ""
                mrz_lines += 1
                mrz_line_exact += actual == expected
                mrz_chars += len(expected) - _distance(expected, actual)
                mrz_char_total += len(expected)
        documents_exact += doc_exact
    return {
        "field_correctness": exact / fields if fields else None,
        "field_character_accuracy": chars / char_total if char_total else None,
        "document_correctness": documents_exact / len(documents) if documents else None,
        "mrz_found_rate": mrz_found / mrz_docs if mrz_docs else None,
        "mrz_exact_match_rate": mrz_full / mrz_docs if mrz_docs else None,
        "mrz_line_accuracy": mrz_line_exact / mrz_lines if mrz_lines else None,
        "mrz_character_accuracy": mrz_chars / mrz_char_total if mrz_char_total else None,
        "field_exact": exact, "field_total": fields, "field_characters": chars, "field_character_total": char_total, "document_exact": documents_exact,
        "mrz_found": mrz_found, "mrz_documents": mrz_docs, "mrz_full_exact": mrz_full,
        "mrz_line_exact": mrz_line_exact, "mrz_lines": mrz_lines, "mrz_characters": mrz_chars, "mrz_character_total": mrz_char_total,
    }


def _stage_totals(payloads: list[dict[str, Any]]) -> dict[str, float]:
    names = ("localization", "pipeline", "text_detection", "text_recognition", "mrz_recognition")
    totals = {name: 0.0 for name in names}
    for payload in payloads:
        diagnostics = payload.get("diagnostics", {})
        for name in names:
            value = diagnostics.get(name, {})
            if name in {"text_detection", "text_recognition", "mrz_recognition"}:
                totals[name] += float(value.get("elapsed_wall_seconds", value.get("wall_seconds", 0.0)))
            else:
                totals[name] += sum(
                    float(call.get("wall_seconds", 0.0))
                    for stage in value.values()
                    if isinstance(stage, dict)
                    for call in stage.get("calls", ())
                ) if name == "localization" else float(value.get("document_preparation_seconds", 0.0)) + float(value.get("mrz_crop_preprocess_seconds", 0.0)) + float(value.get("result_assembly_seconds", 0.0))
    return totals


def _verify_auxiliary_cache(candidate: Candidate) -> None:
    if candidate.category.startswith("docaligner"):
        from importlib import import_module

        module = import_module("docaligner.point_reg.infer" if candidate.env["DOCALIGNER_MODEL_TYPE"] == "point" else "docaligner.heatmap_reg.infer")
        config = module.Inference.configs.get(candidate.env["DOCALIGNER_MODEL"])
        if config is None:
            raise RuntimeError(f"unsupported DocAligner candidate: {candidate.model}")
        path = Path(module.__file__).parent / "ckpt" / config["model_path"]
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f"prepared DocAligner model missing: {path}")
    if candidate.category == "mrz":
        from importlib import import_module

        det_module = import_module("mrzscanner.det.infer")
        det_config = det_module.Inference.configs.get("20250222")
        det_path = Path(det_module.__file__).parent / "ckpt" / det_config["model_path"] if det_config else Path("missing")
        if det_config is None or not det_path.is_file() or not det_path.stat().st_size:
            raise RuntimeError(f"prepared MRZScanner detection model missing: {det_path}")
        kind = "spotting" if candidate.env["MRZ_RECOGNIZER_BACKEND"] == "mrzscanner-spotting" else "rec" if candidate.env["MRZ_RECOGNIZER_BACKEND"] == "mrzscanner" else "det"
        module = import_module(f"mrzscanner.{kind}.infer")
        config_name = candidate.env["MRZ_RECOGNIZER_MODEL"] if kind != "det" else "20250222"
        config = module.Inference.configs.get(config_name)
        if config is None:
            raise RuntimeError(f"unsupported MRZ candidate: {candidate.model}")
        path = Path(module.__file__).parent / "ckpt" / config["model_path"]
        if not path.is_file() or not path.stat().st_size:
            raise RuntimeError(f"prepared MRZScanner model missing: {path}")


def _row(candidate: Candidate, measurements: list[dict[str, Any]], server_info: dict[str, Any], notes: str = "") -> dict[str, Any]:
    scores = [m["score"] for m in measurements]
    latency = [m["latency_seconds"] for m in measurements]
    keys = ("field_correctness", "document_correctness", "mrz_found_rate", "mrz_exact_match_rate", "mrz_line_accuracy", "mrz_character_accuracy")
    result: dict[str, Any] = {"category": candidate.category, "model": candidate.model, "model_type": candidate.env.get("DOCALIGNER_MODEL_TYPE"), "status": "ok", "notes": notes}
    for key in keys:
        result[key] = sum(s[key] for s in scores if s[key] is not None) / len([s for s in scores if s[key] is not None]) if any(s[key] is not None for s in scores) else None
    result.update({"total_latency_seconds": statistics.median(latency), "median_latency_seconds": statistics.median(latency), "throughput_docs_per_second": 20 / statistics.median(latency), "peak_memory_mb": server_info.get("peak_process_memory_mb"), "memory_before_mb": server_info.get("memory_before_mb"), "memory_after_shutdown_mb": server_info.get("memory_after_shutdown_mb"), "cleanup_verified": server_info.get("cleanup_verified")})
    for name in ("localization", "pipeline", "text_detection", "text_recognition", "mrz_recognition"):
        values = [m["stages"].get(name, 0.0) for m in measurements]
        result[f"{name}_seconds"] = statistics.median(values)
    return result


def main() -> int:
    args = _args()
    selected = candidates()
    if args.recognizers:
        wanted = set(args.recognizers)
        selected = [candidate for candidate in selected if candidate.category == "recognizer" and candidate.model in wanted]
        if {candidate.model for candidate in selected} != wanted:
            raise RuntimeError(f"requested recognizers are not available: {sorted(wanted - {candidate.model for candidate in selected})}")
    documents, manifest = validate_and_manifest(args.dataset_root)
    by_kind = {kind: [doc for doc in documents if doc.document_type == kind] for kind in ("passport", "id_card", "driving_license")}
    output = args.output_root / datetime.now().strftime("%Y%m%dT%H%M%S")
    (output / "raw").mkdir(parents=True, exist_ok=True)
    (output / "server_logs").mkdir(parents=True, exist_ok=True)
    (output / "manifest.json").write_text(json.dumps({"dataset": manifest, "candidates": [candidate.__dict__ for candidate in selected], "repeats": args.repeats, "warmup": args.warmup, "cpu_only": True, "server_start": "uvicorn app.main:app --workers 1 (the Makefile's `run` command with explicit MODEL_DIR)", "server_stop": "SIGTERM to the tracked process group; SIGKILL only after timeout"}, indent=2), encoding="utf-8")
    rows: list[dict[str, Any]] = []
    for index, candidate in enumerate(selected, 1):
        print(f"[{index}/{len(selected)}] {candidate.category}: {candidate.model}", flush=True)
        if candidate.blocked:
            rows.append({"category": candidate.category, "model": candidate.model, "model_type": candidate.env.get("DOCALIGNER_MODEL_TYPE"), "status": "blocked", "notes": candidate.blocked})
            continue
        for name in ("TEXT_DETECTOR_MODEL", "TEXT_RECOGNIZER_MODEL"):
            model_path = args.model_dir / "official_models" / candidate.env[name]
            if not all((model_path / file).is_file() and (model_path / file).stat().st_size for file in ("inference.yml", "inference.json", "inference.pdiparams")):
                raise RuntimeError(f"prepared model missing before {candidate.model}: {model_path}")
        _verify_auxiliary_cache(candidate)
        candidate_dir = output / f"{index:02d}_{candidate.category}_{candidate.model.replace('/', '_').replace(' ', '_').replace('+', 'plus') }"
        candidate_dir.mkdir(parents=True, exist_ok=True)
        server = Server(args, candidate_dir, candidate.env)
        memory_before = _available_memory()
        measurements: list[dict[str, Any]] = []
        server_info: dict[str, Any] = {"memory_before_mb": memory_before / 1024 / 1024 if memory_before else None}
        try:
            ready = server.start()
            (candidate_dir / "loaded_configuration.json").write_text(json.dumps(ready, indent=2), encoding="utf-8")
            for repeat in range(args.warmup):
                for kind, docs in by_kind.items():
                    payload, _ = _post(kind, docs, args.port, args.timeout)
                    (candidate_dir / "raw" ).mkdir(exist_ok=True)
                    (candidate_dir / "raw" / f"warmup-{repeat + 1}-{kind}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
            for repeat in range(1, args.repeats + 1):
                started = time.perf_counter()
                payloads, elapsed = [], 0.0
                repeat_dir = candidate_dir / "raw" / f"repeat-{repeat}"
                repeat_dir.mkdir(parents=True, exist_ok=True)
                repeat_scores = []
                for kind, docs in by_kind.items():
                    payload, request_seconds = _post(kind, docs, args.port, args.timeout)
                    payloads.append(payload); elapsed += request_seconds; repeat_scores.append(_score(kind, docs, payload))
                    (repeat_dir / f"{kind}.json").write_text(json.dumps({"client_wall_seconds": request_seconds, "response": payload}, indent=2), encoding="utf-8")
                merged = {key: sum(score[key] for score in repeat_scores if score[key] is not None) for key in repeat_scores[0] if isinstance(repeat_scores[0][key], (int, float))}
                for key in ("field_correctness", "document_correctness", "mrz_found_rate", "mrz_exact_match_rate", "mrz_line_accuracy", "mrz_character_accuracy"):
                    if key == "field_correctness": merged[key] = merged["field_exact"] / merged["field_total"] if merged["field_total"] else None
                    elif key == "document_correctness": merged[key] = merged["document_exact"] / 20
                    elif key == "mrz_found_rate": merged[key] = merged["mrz_found"] / merged["mrz_documents"] if merged["mrz_documents"] else None
                    elif key == "mrz_exact_match_rate": merged[key] = merged["mrz_full_exact"] / merged["mrz_documents"] if merged["mrz_documents"] else None
                    elif key == "mrz_line_accuracy": merged[key] = merged["mrz_line_exact"] / merged["mrz_lines"] if merged["mrz_lines"] else None
                    else: merged[key] = merged["mrz_characters"] / merged["mrz_character_total"] if merged["mrz_character_total"] else None
                measurements.append({"repeat": repeat, "latency_seconds": elapsed, "score": merged, "stages": _stage_totals(payloads), "wall_seconds": time.perf_counter() - started})
                server._sample()
            (candidate_dir / "measurements.json").write_text(json.dumps(measurements, indent=2), encoding="utf-8")
        except Exception as error:
            (candidate_dir / "failure.json").write_text(json.dumps({"error": f"{type(error).__name__}: {error}"}, indent=2), encoding="utf-8")
            raise
        finally:
            cleanup = server.stop()
            memory_after = _available_memory()
            server_info.update(cleanup, memory_after_shutdown_mb=memory_after / 1024 / 1024 if memory_after else None)
            (candidate_dir / "lifecycle.json").write_text(json.dumps(server_info, indent=2), encoding="utf-8")
        if not server_info["cleanup_verified"]:
            raise RuntimeError(f"server cleanup failed for {candidate.model}: {server_info}")
        rows.append(_row(candidate, measurements, server_info))
        print(f"    complete; median {rows[-1]['median_latency_seconds']:.3f}s", flush=True)
    fields = ["category", "model", "model_type", "status", "field_correctness", "document_correctness", "mrz_found_rate", "mrz_exact_match_rate", "mrz_line_accuracy", "mrz_character_accuracy", "median_latency_seconds", "throughput_docs_per_second", "peak_memory_mb", "memory_before_mb", "memory_after_shutdown_mb", "cleanup_verified", "localization_seconds", "pipeline_seconds", "text_detection_seconds", "text_recognition_seconds", "mrz_recognition_seconds", "notes"]
    with (output / "aggregate.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader(); writer.writerows({field: row.get(field) for field in fields} for row in rows)
    (output / "aggregate.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    expected = len(selected)
    if len(rows) != expected:
        raise RuntimeError(f"candidate row count mismatch: {len(rows)} != {expected}")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
