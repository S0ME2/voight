"""Hard server-only guard.  Importing this module performs no checks."""

from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass


class ExecutionRefused(RuntimeError):
    pass


DEFAULT_GPU_MODEL = "Tesla V100-PCIE-32GB"
DEFAULT_DRIVER_VERSION = "535.309.01"


@dataclass(frozen=True)
class GuardResult:
    gpu_id: int
    nvidia_smi: str
    image: str


def require_server_execution(*, execute: bool, image: str = "voight:gpu", gpu_id: int | None = None, runner=subprocess.run) -> GuardResult:
    """Refuse before any Docker launch or GPU library import can occur."""
    if not execute:
        raise ExecutionRefused("real execution requires --execute")
    if os.getenv("RUNTIME_TARGET", "").strip().lower() != "gpu":
        raise ExecutionRefused("RUNTIME_TARGET=gpu is required")
    if os.getenv("VOIGHT_GPU_BENCHMARK_HOST") != "1":
        raise ExecutionRefused("set VOIGHT_GPU_BENCHMARK_HOST=1 on the V100 server")
    if not shutil.which("nvidia-smi"):
        raise ExecutionRefused("nvidia-smi is not available")
    selected = int(gpu_id if gpu_id is not None else os.getenv("GPU_ID", "0"))
    try:
        result = runner(["nvidia-smi", "-i", str(selected), "--query-gpu=index,name,uuid,driver_version", "--format=csv,noheader"], capture_output=True, text=True, check=False, timeout=10)
    except OSError as exc:
        raise ExecutionRefused(f"nvidia-smi failed: {exc}") from exc
    if result.returncode != 0 or not result.stdout.strip():
        raise ExecutionRefused("expected NVIDIA GPU is not visible")
    expected_model = os.getenv("VOIGHT_GPU_MODEL", DEFAULT_GPU_MODEL)
    expected_driver = os.getenv("VOIGHT_GPU_DRIVER_VERSION", DEFAULT_DRIVER_VERSION)
    if expected_model not in result.stdout:
        raise ExecutionRefused(f"expected GPU model is not visible: {expected_model}")
    if expected_driver not in result.stdout:
        raise ExecutionRefused(f"expected NVIDIA driver is not visible: {expected_driver}")
    if not shutil.which("docker"):
        raise ExecutionRefused("docker is not available")
    image_check = runner(["docker", "image", "inspect", image], capture_output=True, text=True, check=False, timeout=10)
    if image_check.returncode != 0:
        raise ExecutionRefused(f"required image is unavailable: {image}")
    return GuardResult(selected, result.stdout.strip(), image)
