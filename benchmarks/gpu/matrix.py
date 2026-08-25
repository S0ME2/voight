"""Pure benchmark matrix expansion and validation.

This module intentionally has no application or GPU imports.  It is safe to
use for planning on a CPU-only workstation.
"""

from __future__ import annotations

import json
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Any

BASELINE: dict[str, Any] = {
    "RUNTIME_TARGET": "gpu",
    "OCR_DEVICE": "gpu",
    "CPU_THREADS": 4,
    "REQUEST_QUEUE_LIMIT": 8,
    "LOCALIZATION_BATCH_SIZE": 4,
    "TEXT_DETECTION_BATCH_SIZE": 1,
    "TEXT_RECOGNITION_BATCH_SIZE": 2,
    "MRZ_RECOGNITION_BATCH_SIZE": 2,
    "TEXT_RECOGNITION_PROCESSES": 1,
    "TEXT_RECOGNITION_PACKING": "fixed-width",
    "TEXT_RECOGNITION_PRECISION": "fp32",
    "TEXT_RECOGNITION_ENABLE_HPI": "false",
    "TEXT_RECOGNITION_USE_TENSORRT": "false",
    "TEXT_DETECTOR_PIXEL_SCALE": 1.0,
    "TEXT_DETECTOR_PREPROCESSING": "original",
    "VISIBLE_RECOGNITION_PREPROCESSING": "original",
    "MRZ_PREPROCESSING": "contrast_1.50",
    "TEXT_DETECTOR_MODEL": "PP-OCRv6_medium_det",
    "TEXT_RECOGNIZER_MODEL": "latin_PP-OCRv5_mobile_rec",
    "DOCALIGNER_MODEL": "fastvit_sa24",
    "DOCALIGNER_MODEL_TYPE": "heatmap",
    "MRZSCANNER_DETECTION_CFG": "20250222",
    "MRZ_RECOGNIZER_BACKEND": "generic-paddle",
    "MRZ_RECOGNIZER_MODEL": "20250221",
}

EXPERIMENT_DEFAULTS: dict[str, tuple[Any, ...]] = {
    "localization-batch": (1, 2, 4, 8, 16, 32),
    "detection-batch": (1, 2, 4, 8, 16, 32),
    "recognition-batch": (1, 2, 4, 8, 16, 32, 64),
    "mrz-batch": (1, 2, 4, 8, 16, 32),
    "precision": ("fp32", "fp16"),
    "backend": ("normal", "hpi", "tensorrt"),
    "recognition-packing": ("fixed-width", "aspect-ratio", "fixed-width-buckets", "best-fit"),
    "detector-resolution": (100, 90, 80, 70, 60, 50),
    "detector-preprocessing": ("original", "grayscale", "contrast_1.25", "clahe_mild", "gamma_0.8"),
    "visible-preprocessing": ("original", "grayscale", "contrast_1.25", "clahe_mild", "gamma_0.8", "sharpen_light"),
    "mrz-preprocessing": ("original", "grayscale", "contrast_1.25", "clahe_mild", "gamma_0.8", "sharpen_light", "otsu"),
    "concurrency": (1, 2, 4, 8),
    "workers": (1, 2),
}

EXPERIMENT_KEYS = {
    "localization-batch": "LOCALIZATION_BATCH_SIZE",
    "detection-batch": "TEXT_DETECTION_BATCH_SIZE",
    "recognition-batch": "TEXT_RECOGNITION_BATCH_SIZE",
    "mrz-batch": "MRZ_RECOGNITION_BATCH_SIZE",
    "precision": "TEXT_RECOGNITION_PRECISION",
    "recognition-packing": "TEXT_RECOGNITION_PACKING",
    "detector-resolution": "TEXT_DETECTOR_PIXEL_SCALE",
    "detector-preprocessing": "TEXT_DETECTOR_PREPROCESSING",
    "visible-preprocessing": "VISIBLE_RECOGNITION_PREPROCESSING",
    "mrz-preprocessing": "MRZ_PREPROCESSING",
    "concurrency": "BENCHMARK_REQUEST_CONCURRENCY",
    "workers": "BENCHMARK_WORKERS",
}

SUPPORTED_PACKING = {"sequential", "aspect-ratio", "fixed-width", "fixed-width-buckets", "best-fit"}
SUPPORTED_PREPROCESSING = {"original", "grayscale", "contrast_1.15", "contrast_1.25", "contrast_1.30", "contrast_1.50", "clahe_mild", "clahe_medium", "gamma_0.8", "gamma_1.2", "sharpen_light", "otsu", "adaptive"}


@dataclass(frozen=True)
class BenchmarkConfig:
    id: str
    env: dict[str, Any]
    axis: str = "baseline"
    value: Any = None
    invalid_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {"id": self.id, "axis": self.axis, "value": self.value, "env": self.env, "invalid_reason": self.invalid_reason}


def _set_value(base: dict[str, Any], key: str, value: Any) -> dict[str, Any]:
    result = dict(base)
    if key == "TEXT_DETECTOR_PIXEL_SCALE":
        result[key] = 1.0
        result["TEXT_DETECTOR_LIMIT_SIDE_LEN"] = aligned_limit(float(value))
    elif key in {"BENCHMARK_REQUEST_CONCURRENCY", "BENCHMARK_WORKERS"}:
        result[key] = int(value)
    else:
        result[key] = value
    return result


def aligned_limit(percent: float, current: int = 960) -> int:
    return max(32, int(percent / 100 * current / 32 + 0.5) * 32)


def validate_env(env: dict[str, Any]) -> str | None:
    positive = ("CPU_THREADS", "REQUEST_QUEUE_LIMIT", "LOCALIZATION_BATCH_SIZE", "TEXT_DETECTION_BATCH_SIZE", "TEXT_RECOGNITION_BATCH_SIZE", "MRZ_RECOGNITION_BATCH_SIZE", "TEXT_RECOGNITION_PROCESSES", "BENCHMARK_REQUEST_CONCURRENCY", "BENCHMARK_WORKERS")
    for key in positive:
        if key in env and int(env[key]) <= 0:
            return f"{key} must be positive"
    if env.get("RUNTIME_TARGET") != "gpu" or env.get("OCR_DEVICE") != "gpu":
        return "GPU benchmark configurations must select RUNTIME_TARGET=gpu and OCR_DEVICE=gpu"
    if int(env.get("TEXT_RECOGNITION_PROCESSES", 1)) != 1:
        return "GPU execution requires TEXT_RECOGNITION_PROCESSES=1"
    if env.get("TEXT_RECOGNITION_PRECISION") not in {"fp32", "fp16"}:
        return "unsupported recognition precision"
    if env.get("TEXT_RECOGNITION_PACKING") not in SUPPORTED_PACKING:
        return "unsupported recognition packing"
    if env.get("TEXT_RECOGNITION_PACKING") == "best-fit" and (int(env.get("TEXT_RECOGNITION_BATCH_SIZE", 2)) != 2 or int(env.get("MRZ_RECOGNITION_BATCH_SIZE", 2)) != 2):
        return "best-fit packing requires TEXT_RECOGNITION_BATCH_SIZE=2"
    for key in ("TEXT_DETECTOR_PREPROCESSING", "VISIBLE_RECOGNITION_PREPROCESSING", "MRZ_PREPROCESSING"):
        if env.get(key) not in SUPPORTED_PREPROCESSING:
            return f"unsupported preprocessing: {env.get(key)}"
    if env.get("TEXT_RECOGNITION_PRECISION") == "fp16" and env.get("TEXT_RECOGNIZER_BACKEND", "paddle") != "paddle":
        return "FP16 is only supported by the Paddle recognizer"
    if env.get("MRZ_RECOGNIZER_BACKEND") != "generic-paddle":
        return "GPU suite baseline requires generic-paddle MRZ recognition"
    return None


def expand(axis: str | None = None, values: list[Any] | None = None, *, base: dict[str, Any] | None = None, full_cartesian: bool = False) -> list[BenchmarkConfig]:
    base_env = {**BASELINE, **(base or {})}
    if not axis:
        reason = validate_env(base_env)
        return [BenchmarkConfig("baseline", base_env, invalid_reason=reason)]
    if axis not in EXPERIMENT_DEFAULTS:
        raise ValueError(f"unknown experiment {axis!r}; choose from {', '.join(EXPERIMENT_DEFAULTS)}")
    raw_values = values or list(EXPERIMENT_DEFAULTS[axis])
    configs = []
    key = EXPERIMENT_KEYS.get(axis)
    for index, value in enumerate(raw_values, 1):
        env = _set_value(base_env, key, value) if key else base_env
        if axis == "precision" and value == "fp16":
            env["TEXT_RECOGNITION_PRECISION"] = "fp16"
        if axis == "backend":
            env["TEXT_RECOGNITION_ENABLE_HPI"] = "true" if value == "hpi" else "false"
            env["TEXT_RECOGNITION_USE_TENSORRT"] = "true" if value == "tensorrt" else "false"
        reason = validate_env(env)
        configs.append(BenchmarkConfig(f"{axis}-{index:02d}-{str(value).replace('/', '_')}", env, axis, value, reason))
    return configs


def from_json(path: str, *, full_cartesian: bool = False) -> list[BenchmarkConfig]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if full_cartesian and isinstance(value, dict) and isinstance(value.get("matrix"), dict):
        base = {**BASELINE, **value.get("base", {})}
        keys = list(value["matrix"])
        result = []
        for index, values in enumerate(itertools.product(*(value["matrix"][key] for key in keys)), 1):
            env = {**base, **dict(zip(keys, values))}
            result.append(BenchmarkConfig(f"cartesian-{index:03d}", env, "custom", dict(zip(keys, values)), validate_env(env)))
        return result
    rows = value if isinstance(value, list) else [value]
    result = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row, dict):
            raise ValueError("custom configuration must be an object or list of objects")
        env = {**BASELINE, **row.get("env", row)}
        result.append(BenchmarkConfig(row.get("id", f"custom-{index:02d}"), env, "custom", row.get("value"), validate_env(env)))
    return result


def estimate(configs: list[BenchmarkConfig], logical_documents: int, *, repeats: int, warmups: int) -> dict[str, int]:
    warmup_requests = sum(warmups * logical_documents * int(config.env.get("BENCHMARK_REQUEST_CONCURRENCY", 1)) for config in configs)
    measured_requests = sum(repeats * logical_documents * int(config.env.get("BENCHMARK_REQUEST_CONCURRENCY", 1)) for config in configs)
    return {
        "container_launches": len(configs),
        "warmup_requests": warmup_requests,
        "measured_requests": measured_requests,
        "total_requests": warmup_requests + measured_requests,
        "configuration_count": len(configs),
    }
