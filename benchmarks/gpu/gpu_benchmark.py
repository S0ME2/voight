"""Reusable Voight GPU benchmark runner.

Planning and comparison are CPU-safe.  GPU/server actions are isolated behind
``runtime_guard.require_server_execution`` and are only reached by ``--execute``.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import subprocess
import sys
import time
import uuid
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from urllib import request

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.gpu.collectors import GPUSampler, system_metadata
from benchmarks.gpu.helpers import compare_semantics, correctness_score, dataset_manifest, semantic_digest
from benchmarks.gpu.matrix import BASELINE, EXPERIMENT_DEFAULTS, expand, estimate, from_json
from benchmarks.gpu.reporting import summary_rows, write_report
from benchmarks.gpu.runtime_guard import require_server_execution

ROOT = Path(__file__).resolve().parents[2]


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan", action="store_true", help="expand and print a CPU-safe plan")
    mode.add_argument("--execute", action="store_true", help="run on an acknowledged V100 server")
    parser.add_argument("--mode", choices=("smoke", "baseline", "experiment", "custom"), default="baseline")
    parser.add_argument("--experiment", choices=tuple(EXPERIMENT_DEFAULTS), help="one staged benchmark axis")
    parser.add_argument("--values", help="comma-separated values for --experiment")
    parser.add_argument("--config", type=Path, help="JSON object/list for --mode custom")
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/benchmarks/gpu")
    parser.add_argument("--image", default="voight:gpu")
    parser.add_argument("--gpu-id", type=int, default=int(os.getenv("GPU_ID", "0")))
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--request-concurrency", type=int, default=1)
    parser.add_argument("--full-cartesian", action="store_true", help="reserved for explicit custom matrices")
    parser.add_argument("--json", action="store_true", dest="json_output", help="emit the complete machine-readable plan")
    args = parser.parse_args(argv)
    if args.repeats < 3 or args.warmup < 1 or args.port <= 0 or args.request_concurrency <= 0:
        parser.error("warmup >= 1, repeats >= 3, port > 0, and concurrency > 0 are required")
    if args.mode == "custom" and not args.config:
        parser.error("--mode custom requires --config JSON")
    if args.mode == "experiment" and not args.experiment:
        parser.error("--mode experiment requires --experiment")
    if args.mode == "smoke":
        args.warmup, args.repeats = 1, 3
    return args


def _value_list(axis: str | None, raw: str | None) -> list[object] | None:
    if not raw:
        return None
    values = []
    for value in raw.split(","):
        value = value.strip()
        if axis in {"precision", "backend", "recognition-packing", "detector-preprocessing", "visible-preprocessing", "mrz-preprocessing"}:
            values.append(value)
        elif axis == "detector-resolution":
            values.append(float(value))
        else:
            values.append(int(value))
    return values


def build_configs(args: argparse.Namespace):
    if args.mode == "custom":
        return from_json(str(args.config), full_cartesian=args.full_cartesian)
    axis = args.experiment if args.experiment else None
    configs = expand(axis, _value_list(axis, args.values))
    if args.mode == "smoke":
        configs[0] = configs[0].__class__("smoke", configs[0].env, "smoke", None, configs[0].invalid_reason)
    for config in configs:
        config.env["BENCHMARK_REQUEST_CONCURRENCY"] = args.request_concurrency
    return configs


def plan(args: argparse.Namespace) -> int:
    documents, manifest = dataset_manifest(args.dataset_root)
    configs = build_configs(args)
    estimate_data = estimate(configs, len({document["document_type"] for document in documents}), repeats=args.repeats, warmups=args.warmup)
    payload = {"mode": args.mode, "experiment": args.experiment, "dataset": manifest, "configs": [config.as_dict() for config in configs], "estimate": estimate_data}
    if args.json_output:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    else:
        print("Voight GPU benchmark plan")
        print(f"mode={args.mode} experiment={args.experiment or 'baseline'}")
        print(f"dataset={len(documents)} logical documents, {manifest['physical_images']} physical images; order=passport,id_card,driving_license")
        print("resolved base: " + ", ".join(f"{key}={value}" for key, value in BASELINE.items()))
        print("configurations:")
        for index, config in enumerate(configs, 1):
            changes = {key: value for key, value in config.env.items() if BASELINE.get(key) != value}
            if config.axis == "custom" and isinstance(config.value, dict):
                changes = dict(config.value)
            status = f"INVALID: {config.invalid_reason}" if config.invalid_reason else "ready"
            change_text = ", ".join(f"{key}={value}" for key, value in changes.items()) or "baseline values"
            print(f"  {index:02d}. {config.id}: {change_text} [{status}]")
        print("estimate:")
        for key, value in estimate_data.items():
            print(f"  {key.replace('_', ' ')}: {value}")
    return 0 if not any(config.invalid_reason for config in configs) else 2


def _multipart(fields: list[tuple[str, str, bytes, str]]) -> tuple[bytes, str]:
    boundary = f"voight-{uuid.uuid4().hex}"
    body = io.BytesIO()
    for name, filename, data, content_type in fields:
        body.write(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: {content_type}\r\n\r\n".encode())
        body.write(data)
        body.write(b"\r\n")
    body.write(f"--{boundary}--\r\n".encode())
    return body.getvalue(), f"multipart/form-data; boundary={boundary}"


def _request_payload(kind: str, documents: list[dict[str, object]], port: int, timeout: float) -> tuple[dict, float]:
    fields: list[tuple[str, str, bytes, str]] = []
    if kind == "id_card":
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
            for document in documents:
                for role, path in document["paths"]:  # type: ignore[index]
                    path = Path(path)
                    zipped.writestr(f"{document['document_id']}/{role}{path.suffix.lower()}", path.read_bytes())
        fields = [("archive", "cards.zip", archive.getvalue(), "application/zip")]
    else:
        for document in documents:
            role, path = document["paths"][0]  # type: ignore[index]
            path = Path(path)
            fields.append(("images", f"{document['document_id']}{path.suffix.lower()}", path.read_bytes(), "image/png" if path.suffix.lower() == ".png" else "image/jpeg"))
    body, content_type = _multipart(fields)
    started = time.perf_counter()
    http_request = request.Request(f"http://127.0.0.1:{port}/v1/ocr/{kind.replace('_', '-')}/batch", data=body, headers={"Content-Type": content_type}, method="POST")
    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            payload = json.loads(response.read())
    except Exception as exc:
        return {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "items": [], "failed": len(documents)}, time.perf_counter() - started
    return payload, time.perf_counter() - started


def _ready(port: int, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last = "not started"
    while time.monotonic() < deadline:
        try:
            with request.urlopen(f"http://127.0.0.1:{port}/v1/health/ready", timeout=5) as response:
                return json.loads(response.read())
        except Exception as exc:
            last = str(exc)
            time.sleep(1)
    raise RuntimeError(f"server readiness timed out: {last}")


def _docker(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, check=check, timeout=120)


def _container_memory_mb(name: str) -> float | None:
    result = _docker(["docker", "stats", "--no-stream", "--format", "{{.MemUsage}}", name], check=False)
    match = re.search(r"([0-9.]+)\s*([KMG]i?B)", result.stdout)
    if not match:
        return None
    scale = {"KB": 1 / 1024, "KiB": 1 / 1024, "MB": 1, "MiB": 1, "GB": 1024, "GiB": 1024}
    return float(match.group(1)) * scale[match.group(2)]


def _verify_ready(ready: dict, expected: dict[str, object]) -> None:
    models = ready.get("models", {})
    runtime = ready.get("runtime", {})
    if runtime.get("target") != "gpu":
        raise RuntimeError(f"server did not report GPU runtime: {runtime}")
    checks = {"text_detector": ("model", expected["TEXT_DETECTOR_MODEL"]), "text_recognizer": ("model", expected["TEXT_RECOGNIZER_MODEL"]), "document_localizer": ("model_cfg", expected["DOCALIGNER_MODEL"]), "mrz_localizer": ("model_cfg", expected["MRZSCANNER_DETECTION_CFG"]), "mrz_recognizer": ("backend", expected["MRZ_RECOGNIZER_BACKEND"])}
    for section, (key, value) in checks.items():
        actual = models.get(section, {})
        if not actual.get("loaded") or actual.get(key) != value:
            raise RuntimeError(f"loaded configuration mismatch for {section}.{key}: expected {value!r}, got {actual}")
    if models.get("document_localizer", {}).get("model_type") != expected["DOCALIGNER_MODEL_TYPE"]:
        raise RuntimeError("loaded document localizer model type does not match request")
    acceleration = models.get("recognition_acceleration", {})
    for key, expected_value in {
        "precision": expected["TEXT_RECOGNITION_PRECISION"],
        "hpi": str(expected["TEXT_RECOGNITION_ENABLE_HPI"]).lower() == "true",
        "tensorrt": str(expected["TEXT_RECOGNITION_USE_TENSORRT"]).lower() == "true",
    }.items():
        if acceleration.get(key) != expected_value:
            raise RuntimeError(f"effective recognition acceleration mismatch for {key}: expected {expected_value!r}, got {acceleration.get(key)!r}")
    batches = models.get("batch", {})
    for key, env_key in {"localization": "LOCALIZATION_BATCH_SIZE", "detection": "TEXT_DETECTION_BATCH_SIZE", "recognition": "TEXT_RECOGNITION_BATCH_SIZE", "mrz_recognition": "MRZ_RECOGNITION_BATCH_SIZE"}.items():
        if batches.get(key) != int(expected[env_key]):
            raise RuntimeError(f"effective batch mismatch for {key}: expected {expected[env_key]!r}, got {batches.get(key)!r}")
    if models.get("recognition_packing") != expected["TEXT_RECOGNITION_PACKING"]:
        raise RuntimeError("effective recognition packing does not match request")
    if models.get("detector_preprocessing") != expected["TEXT_DETECTOR_PREPROCESSING"] or models.get("visible_recognition_preprocessing") != expected["VISIBLE_RECOGNITION_PREPROCESSING"] or models.get("mrz_preprocessing", {}).get("variant") != expected["MRZ_PREPROCESSING"]:
        raise RuntimeError("effective preprocessing does not match request")
    resize = models.get("text_detector", {}).get("resize", {})
    if abs(float(resize.get("pixel_scale", 1.0)) - float(expected["TEXT_DETECTOR_PIXEL_SCALE"])) > 1e-6:
        raise RuntimeError("effective detector resolution does not match request")
    expected_limit = expected.get("TEXT_DETECTOR_LIMIT_SIDE_LEN")
    if expected_limit is not None and int(resize.get("effective_limit_side_len")) != int(expected_limit):
        raise RuntimeError("effective detector limit side does not match request")


def _run_config(config, args, documents, output: Path, sampler: GPUSampler, references: dict[str, dict] | None = None) -> tuple[list[dict], dict, dict, list[dict], dict[str, dict]]:
    name = f"voight-gpu-bench-{uuid.uuid4().hex[:10]}"
    env = {key: str(value).lower() if isinstance(value, bool) else str(value) for key, value in config.env.items() if not key.startswith("BENCHMARK_")}
    env.update({"RUNTIME_TARGET": "gpu", "OCR_DEVICE": "gpu", "PRELOAD": "true", "LOGGING": "false", "GPU_ID": str(args.gpu_id)})
    command = ["docker", "run", "-d", "--rm", "--name", name, "--gpus", f"device={args.gpu_id}", "-p", f"{args.port}:8000"]
    for key, value in env.items():
        command.extend(("-e", f"{key}={value}"))
    command.append(args.image)
    workers = int(config.env.get("BENCHMARK_WORKERS", 1))
    if workers != 1:
        command.extend(("uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", str(workers)))
    started = time.time()
    lifecycle = {"config_id": config.id, "container": name, "started_at": started, "cleanup_verified": False}
    rows: list[dict] = []
    responses: dict[str, dict] = {}
    try:
        _docker(command)
        ready = _ready(args.port, args.timeout)
        _verify_ready(ready, config.env)
        versions = _docker(["docker", "exec", name, "python", "-c", "m=__import__; p=m('paddle'); o=m('onnxruntime'); u=m('importlib.util', fromlist=['find_spec']); t=m('tensorrt').__version__ if u.find_spec('tensorrt') else None; print({'paddle': p.__version__, 'onnxruntime': o.__version__, 'tensorrt': t})"], check=False)
        lifecycle["container_runtime_versions"] = versions.stdout.strip() or versions.stderr.strip()
        (output / f"{config.id}.ready.json").write_text(json.dumps(ready, indent=2), encoding="utf-8")
        sampler.start()
        kinds = ("passport", "id_card", "driving_license")
        for phase, repeat_count in (("warmup", args.warmup), ("measured", args.repeats)):
            sampler.phase = phase
            for repeat in range(1, repeat_count + 1):
                for kind in kinds:
                    selected = [document for document in documents if document["document_type"] == kind]
                    with ThreadPoolExecutor(max_workers=int(config.env.get("BENCHMARK_REQUEST_CONCURRENCY", 1))) as pool:
                        values = list(pool.map(lambda _unused: _request_payload(kind, selected, args.port, args.timeout), range(int(config.env.get("BENCHMARK_REQUEST_CONCURRENCY", 1)))))
                    aggregate_throughput = (len(selected) * len(values) / max((latency for _, latency in values), default=0.0)) if values else None
                    for request_index, (payload, latency) in enumerate(values):
                        digest = semantic_digest(payload)
                        baseline = responses.get(kind) or (references or {}).get(kind)
                        differences = compare_semantics(baseline, payload) if baseline else {"semantic_change_count": 0, "confidence_only_change_count": 0}
                        if phase == "measured" and baseline is None:
                            responses[kind] = payload
                        diagnostics = payload.get("diagnostics", {})
                        correctness = correctness_score(selected, payload)
                        measured_samples = [sample for sample in sampler.rows if sample.get("phase") == "measured"]
                        gpu_rows = measured_samples[-1:] if measured_samples else []
                        rows.append({"config_id": config.id, "axis": config.axis, "value": config.value, "phase": phase, "repeat": repeat, "request_index": request_index, "document_type": kind, "status": "ok" if payload.get("failed", 0) == 0 and payload.get("status") != "failed" else "failed", "latency_seconds": latency, "throughput_per_second": len(selected) / latency if latency else None, "aggregate_throughput_per_second": aggregate_throughput, "server_seconds": payload.get("total_seconds"), "failures": payload.get("failed", 0), "host_rss_mb": diagnostics.get("process_peak_rss_mb"), "semantic_digest": digest, "semantic_changes": differences.get("semantic_changes", []), "confidence_only_changes": differences.get("confidence_only_changes", []), "semantic_change_count": differences.get("semantic_change_count", 0), "confidence_only_change_count": differences.get("confidence_only_change_count", 0), "correctness": correctness, "configured_batches": {name: diagnostics.get(name, {}).get("configured_batch_size") for name in ("localization", "text_detection", "text_recognition", "mrz_recognition")}, "actual_tensor_batches": {name: diagnostics.get(name, {}).get("tensor_batch_sizes", []) for name in ("localization", "text_detection", "text_recognition", "mrz_recognition")}, "preprocessing_seconds": diagnostics.get("preprocessing_seconds", {}), "localization_calls": diagnostics.get("localization", {}), "detector_calls": diagnostics.get("text_detection", {}).get("calls", []), "recognition_calls": diagnostics.get("text_recognition", {}).get("calls", []), "mrz_recognition_calls": diagnostics.get("mrz_recognition", {}).get("calls", []), "detector_tensor_shapes": diagnostics.get("text_detection", {}).get("calls", []), "peak_vram_mb": max((sample.get("memory_used_mb") for sample in measured_samples if sample.get("memory_used_mb") is not None), default=None), "peak_gpu_utilization_percent": max((sample.get("gpu_utilization_percent") for sample in measured_samples if sample.get("gpu_utilization_percent") is not None), default=None), "gpu_sample": gpu_rows, "response": payload})
                        if phase == "measured" and baseline is None:
                            responses[kind] = payload
    finally:
        sampler.stop()
        lifecycle["container_memory_mb"] = _container_memory_mb(name)
        stopped = _docker(["docker", "stop", name], check=False)
        remaining = _docker(["docker", "ps", "-q", "--filter", f"name=^{name}$"], check=False)
        lifecycle.update({"stopped_returncode": stopped.returncode, "remaining_container": remaining.stdout.strip(), "cleanup_verified": not remaining.stdout.strip(), "stopped_at": time.time()})
    if not lifecycle["cleanup_verified"]:
        raise RuntimeError(f"container cleanup failed: {lifecycle}")
    for row in rows:
        row["cleanup_verified"] = lifecycle["cleanup_verified"]
        row["container_memory_mb"] = lifecycle["container_memory_mb"]
    return rows, lifecycle, ready, list(sampler.rows), responses


def execute(args: argparse.Namespace) -> int:
    # This is deliberately the first execution-side call.  It must remain
    # before Docker, nvidia-smi sampling, or any optional runtime import.
    guard = require_server_execution(execute=True, image=args.image, gpu_id=args.gpu_id)
    documents, manifest = dataset_manifest(args.dataset_root)
    configs = build_configs(args)
    invalid = [config for config in configs if config.invalid_reason]
    if invalid:
        raise SystemExit("invalid configuration: " + "; ".join(f"{config.id}: {config.invalid_reason}" for config in invalid))
    output = args.output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    system = {**system_metadata(args.gpu_id), "guard": guard.__dict__, "docker_version": _docker(["docker", "version", "--format", "{{.Server.Version}}"], check=False).stdout.strip(), "image_id": _docker(["docker", "image", "inspect", args.image, "--format", "{{.Id}}"], check=False).stdout.strip(), "git_sha": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain"))}
    all_rows, lifecycles, semantics, all_gpu_samples, references = [], [], [], [], {}
    workload = documents if args.mode != "smoke" else [document for kind in ("passport", "id_card", "driving_license") if (document := next((item for item in documents if item["document_type"] == kind), None))]
    for config in configs:
        config_sampler = GPUSampler(output / "gpu_samples.csv", args.gpu_id)
        rows, lifecycle, ready, gpu_samples, observed = _run_config(config, args, workload, output, config_sampler, references)
        all_rows.extend(rows)
        lifecycles.append(lifecycle)
        for sample in gpu_samples:
            sample["config_id"] = config.id
        all_gpu_samples.extend(gpu_samples)
        for kind, payload in observed.items():
            references.setdefault(kind, payload)
        semantics.append({"config_id": config.id, "semantic_digests": [row["semantic_digest"] for row in rows if row["phase"] == "measured"], "differences": [{"document_type": row["document_type"], "repeat": row["repeat"], "request_index": row["request_index"], "semantic_changes": row["semantic_changes"], "confidence_only_changes": row["confidence_only_changes"]} for row in rows if row["phase"] == "measured" and (row["semantic_changes"] or row["confidence_only_changes"])], "ready": ready})
    system["container_runtime_versions"] = lifecycles[0].get("container_runtime_versions") if lifecycles else None
    (output / "system.json").write_text(json.dumps(system, indent=2), encoding="utf-8")
    write_report(output, [config.as_dict() for config in configs], all_rows, lifecycles, all_gpu_samples, semantics, system=system, experiment={"mode": args.mode, "experiment": args.experiment, "warmup": args.warmup, "repeats": args.repeats, "smoke_workload": bool(args.mode == "smoke")}, dataset=manifest)
    print(f"completed: {output}")
    for row in summary_rows(all_rows, all_gpu_samples):
        print(
            "  {config_id}/{document_type}: median_latency={latency_seconds}s "
            "aggregate_throughput={aggregate_throughput_per_second} docs/s "
            "peak_vram={peak_vram_mb}MB median_gpu_util={median_gpu_utilization_percent}% "
            "correctness={field_correctness} semantic_changes={semantic_change_count} "
            "cleanup={cleanup_verified}".format(**row)
        )
    return 0


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)
    return result.stdout.strip()


def main(argv: list[str] | None = None) -> int:
    args = arguments(argv)
    return plan(args) if args.plan else execute(args)


if __name__ == "__main__":
    raise SystemExit(main())
