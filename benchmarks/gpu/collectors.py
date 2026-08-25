"""GPU/host collectors; only called after the explicit execution guard."""

from __future__ import annotations

import csv
import os
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Callable


def parse_nvidia_smi(text: str) -> list[dict[str, Any]]:
    rows = []
    for row in csv.reader(line for line in text.splitlines() if line.strip()):
        if len(row) < 8:
            continue
        values = [value.strip() for value in row]
        def number(index: int) -> float | None:
            try:
                return float(values[index])
            except (ValueError, IndexError):
                return None
        rows.append({"gpu_index": values[0], "name": values[1], "uuid": values[2], "memory_used_mb": number(3), "gpu_utilization_percent": number(4), "memory_utilization_percent": number(5), "temperature_c": number(6), "power_draw_w": number(7), "timestamp": time.time()})
    return rows


def nvidia_sample(gpu_id: int = 0, runner=subprocess.run) -> list[dict[str, Any]]:
    result = runner(["nvidia-smi", "-i", str(gpu_id), "--query-gpu=index,name,uuid,memory.used,utilization.gpu,utilization.memory,temperature.gpu,power.draw", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False, timeout=10)
    return parse_nvidia_smi(result.stdout) if result.returncode == 0 else []


def host_rss_mb() -> float | None:
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    except (FileNotFoundError, PermissionError, ValueError):
        return None
    return None


def host_cpu_percent() -> float | None:
    try:
        return os.getloadavg()[0] / max(1, os.cpu_count() or 1) * 100
    except (AttributeError, OSError):
        return None


class GPUSampler:
    def __init__(self, path: Path, gpu_id: int, interval: float = 0.25, sample: Callable[[int], list[dict[str, Any]]] = nvidia_sample):
        self.path, self.gpu_id, self.interval, self.sample = path, gpu_id, interval, sample
        self.phase = "unattributed"
        self.rows: list[dict[str, Any]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="voight-gpu-sampler", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            for row in self.sample(self.gpu_id):
                row["phase"] = self.phase
                row["host_rss_mb"] = host_rss_mb()
                row["host_cpu_percent"] = host_cpu_percent()
                self.rows.append(row)
            self._stop.wait(self.interval)

    def stop(self) -> list[dict[str, Any]]:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=max(1.0, self.interval * 4))
        return list(self.rows)


def system_metadata(gpu_id: int = 0) -> dict[str, Any]:
    ram_mb = None
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                ram_mb = int(line.split()[1]) / 1024
                break
    except (FileNotFoundError, PermissionError, ValueError):
        pass
    return {"os": platform.platform(), "python": platform.python_version(), "cpu": platform.processor(), "cpu_count": os.cpu_count(), "ram_mb": ram_mb, "gpu_id": gpu_id, "timestamp": time.time()}
