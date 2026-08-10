"""Lazy runtime/provider readiness checks shared by CPU and GPU deployments."""

from __future__ import annotations

from typing import Any

from app.config import Settings


def validate_runtime(
    settings: Settings,
    *,
    paddle_module: Any | None = None,
    ort_module: Any | None = None,
    localizers: tuple[Any, ...] = (),
) -> dict[str, Any]:
    if paddle_module is None:
        import paddle as paddle_module
    if ort_module is None:
        import onnxruntime as ort_module

    target = settings.runtime.target
    expected_provider = "CPUExecutionProvider" if target == "cpu" else "CUDAExecutionProvider"
    available = list(ort_module.get_available_providers())
    if expected_provider not in available:
        raise RuntimeError(f"ONNX Runtime provider is unavailable: {expected_provider}")
    if target == "gpu":
        if not paddle_module.device.is_compiled_with_cuda():
            raise RuntimeError("Paddle was not built with CUDA")
        if paddle_module.device.cuda.device_count() <= settings.runtime.gpu_id:
            raise RuntimeError(f"configured GPU_ID {settings.runtime.gpu_id} is unavailable")
    for localizer in localizers:
        if expected_provider not in localizer.providers:
            raise RuntimeError(
                f"initialized localization model is not using {expected_provider}"
            )
    return {
        "target": target,
        "gpu_id": settings.runtime.gpu_id,
        "onnx_provider": expected_provider,
        "paddle_device": "cpu" if target == "cpu" else f"gpu:{settings.runtime.gpu_id}",
    }
