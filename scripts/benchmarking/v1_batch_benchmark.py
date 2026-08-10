"""Benchmark only the production /v1 batch routes and their model diagnostics."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import threading
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import requests


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--passport", type=Path, required=True)
    parser.add_argument("--id-front", type=Path, required=True)
    parser.add_argument("--id-back", type=Path, required=True)
    parser.add_argument("--driving-license", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--monitor-interval", type=float, default=1.0)
    parser.add_argument("--no-gpu-monitor", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("outputs/v1-batch-benchmark.json"))
    args = parser.parse_args()
    paths = (args.passport, args.id_front, args.id_back, args.driving_license)
    if args.repeats < 1 or any(size < 1 for size in args.sizes) or args.monitor_interval <= 0:
        parser.error("--repeats and --sizes must be positive")
    if any(not path.is_file() for path in paths):
        parser.error("all sample paths must exist")
    return args


def _id_archive(front: bytes, back: bytes, count: int) -> bytes:
    output = BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        for index in range(count):
            archive.writestr(f"card-{index:03d}/front.jpg", front)
            archive.writestr(f"card-{index:03d}/back.jpg", back)
    return output.getvalue()


def _resource_exhausted(response: requests.Response) -> bool:
    return response.status_code >= 500 and any(
        marker in response.text.lower()
        for marker in ("out of memory", "resource exhausted", "resource_exhausted", "cuda error", "cudnn_status_alloc_failed")
    )


def _post(args: argparse.Namespace, kind: str, size: int) -> dict[str, Any]:
    url = f"{args.base_url.rstrip('/')}/v1/ocr/{kind}/batch"
    if kind == "id-card":
        files: Any = {
            "archive": (
                "cards.zip",
                _id_archive(args.id_front.read_bytes(), args.id_back.read_bytes(), size),
                "application/zip",
            )
        }
    else:
        path = args.passport if kind == "passport" else args.driving_license
        data = path.read_bytes()
        files = [("images", (f"{index}-{path.name}", data, "image/jpeg")) for index in range(size)]
    started = time.perf_counter()
    try:
        response = requests.post(url, files=files, timeout=args.timeout)
    except requests.RequestException as error:
        return {
            "document_type": kind,
            "document_count": size,
            "status": "request_failed",
            "failure_detail": f"{type(error).__name__}: {error}",
        }
    wall = time.perf_counter() - started
    if not response.ok:
        return {
            "document_type": kind,
            "document_count": size,
            "client_wall_seconds": wall,
            "status": "resource_exhausted" if _resource_exhausted(response) else "request_failed",
            "failure_detail": response.text[:1000],
        }
    payload = response.json()
    diagnostics = payload.get("diagnostics", {})
    return {
        "document_type": kind,
        "document_count": size,
        "client_wall_seconds": wall,
        "server_wall_seconds": payload["total_seconds"],
        "documents_per_second": size / wall,
        "succeeded": payload["succeeded"],
        "failures": payload["failed"],
        "status": "ok",
        "localization": diagnostics.get("localization", {}),
        "text_detection": diagnostics.get("text_detection", {}),
        "text_recognition": diagnostics.get("text_recognition", {}),
    }


class GpuMonitor:
    """Use nvidia-smi only; unavailable monitoring is recorded, not hidden."""

    def __init__(self, index: int, interval: float):
        self.index, self.interval = index, interval
        self.samples: list[dict[str, Any]] = []
        self.error: str | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        self._thread.join(timeout=self.interval + 2)
        peak_memory = max((sample["memory_used_mib"] for sample in self.samples), default=None)
        peak_utilization = max((sample["utilization_percent"] for sample in self.samples), default=None)
        return {"samples": self.samples, "peak_memory_mib": peak_memory, "peak_utilization_percent": peak_utilization, "error": self.error}

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                output = subprocess.check_output(
                    ["nvidia-smi", f"--id={self.index}", "--query-gpu=name,utilization.gpu,memory.used", "--format=csv,noheader,nounits"],
                    text=True,
                    stderr=subprocess.STDOUT,
                ).strip()
                name, utilization, memory = (part.strip() for part in output.splitlines()[0].split(","))
                self.samples.append({"timestamp": time.time(), "gpu_name": name, "utilization_percent": int(utilization), "memory_used_mib": int(memory)})
            except (OSError, subprocess.CalledProcessError, ValueError, IndexError) as error:
                self.error = str(error)
                return
            self._stop.wait(self.interval)


def _summary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] == "ok":
            grouped.setdefault((row["document_type"], row["document_count"]), []).append(row)
    return [
        {
            "document_type": kind,
            "request_documents": size,
            "repetitions": len(values),
            "median_seconds": sorted(row["client_wall_seconds"] for row in values)[len(values) // 2],
            "median_documents_per_second": sorted(row["documents_per_second"] for row in values)[len(values) // 2],
            "localization_batches": values[0]["localization"],
            "detection_batches": values[0]["text_detection"],
            "recognition_batches": values[0]["text_recognition"],
        }
        for (kind, size), values in grouped.items()
    ]


def main() -> None:
    args = _args()
    monitor = None if args.no_gpu_monitor else GpuMonitor(args.gpu_index, args.monitor_interval)
    if monitor:
        monitor.start()
    try:
        rows = [
            _post(args, kind, size)
            for size in args.sizes
            for kind in ("passport", "id-card", "driving-license")
            for _ in range(args.repeats)
        ]
    finally:
        monitoring = None if monitor is None else monitor.stop()
    report = {
        "routes": [
            "/v1/ocr/passport/batch",
            "/v1/ocr/id-card/batch",
            "/v1/ocr/driving-license/batch",
        ],
        "rows": rows,
        "summary": _summary(rows),
        "gpu_monitoring": monitoring,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=(
                "document_type",
                "document_count",
                "client_wall_seconds",
                "server_wall_seconds",
                "documents_per_second",
                "succeeded",
                "failures",
                "status",
            ),
        )
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in writer.fieldnames} for row in rows)
    print(args.output)


if __name__ == "__main__":
    main()
