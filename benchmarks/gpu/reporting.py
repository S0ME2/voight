"""Artifact writing and small decision-useful SVG charts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .helpers import csv_write, jsonl_write, stats


def _svg(rows: list[dict[str, Any]], path: Path) -> None:
    points = []
    for index, row in enumerate(rows):
        latency = row.get("latency_seconds")
        throughput = row.get("throughput_per_second")
        if latency is not None and throughput is not None:
            points.append((50 + index * 70, 250 - min(220, float(throughput) * 20), str(row.get("config_id", index))))
    labels = " ".join(f'<text x="{x}" y="280" font-size="10">{label}</text>' for x, _, label in points)
    circles = " ".join(f'<circle cx="{x}" cy="{y}" r="4" fill="#1769aa"/>' for x, y, _ in points)
    path.write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="300"><rect width="100%" height="100%" fill="white"/><text x="20" y="20">Throughput (higher is better) by configuration</text>{circles}{labels}</svg>\n', encoding="utf-8")


def write_report(directory: Path, configs: list[dict[str, Any]], raw_rows: list[dict[str, Any]], lifecycle: list[dict[str, Any]], gpu_samples: list[dict[str, Any]], semantic: list[dict[str, Any]], *, system: dict[str, Any], experiment: dict[str, Any], dataset: dict[str, Any]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "system.json").write_text(json.dumps(system, indent=2, ensure_ascii=False), encoding="utf-8")
    (directory / "experiment.json").write_text(json.dumps(experiment, indent=2, ensure_ascii=False), encoding="utf-8")
    (directory / "configs.json").write_text(json.dumps(configs, indent=2, ensure_ascii=False), encoding="utf-8")
    jsonl_write(directory / "raw_results.jsonl", raw_rows)
    csv_write(directory / "raw_results.csv", raw_rows)
    jsonl_write(directory / "lifecycle.jsonl", lifecycle)
    csv_write(directory / "gpu_samples.csv", gpu_samples)
    csv_write(directory / "comparison.csv", summary_rows(raw_rows, gpu_samples))
    (directory / "semantic_differences.json").write_text(json.dumps(semantic, indent=2, ensure_ascii=False), encoding="utf-8")
    (directory / "manifest.json").write_text(json.dumps({"dataset": dataset, "required_files": ["system.json", "experiment.json", "configs.json", "raw_results.jsonl", "raw_results.csv", "comparison.csv", "semantic_differences.json", "lifecycle.jsonl", "gpu_samples.csv"], "fresh_container_per_configuration": True}, indent=2), encoding="utf-8")
    (directory / "README.txt").write_text("GPU benchmark artifacts. Compare median latency/throughput only after semantic differences and cleanup_verified are reviewed.\n", encoding="utf-8")
    (directory / "plots").mkdir(exist_ok=True)
    _svg(summary_rows(raw_rows, gpu_samples), directory / "plots" / "throughput_by_configuration.svg")


def summary_rows(raw_rows: list[dict[str, Any]], gpu_samples: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in raw_rows:
        if row.get("phase") == "measured":
            grouped.setdefault((str(row.get("config_id")), str(row.get("document_type", "all"))), []).append(row)
    result = []
    for (config_id, document_type), rows in grouped.items():
        latency = stats(row["latency_seconds"] for row in rows if row.get("latency_seconds") is not None)
        throughput = stats(row["throughput_per_second"] for row in rows if row.get("throughput_per_second") is not None)
        aggregate_throughput = stats(row["aggregate_throughput_per_second"] for row in rows if row.get("aggregate_throughput_per_second") is not None)
        correctness = [row.get("correctness", {}).get("field_correctness") for row in rows if isinstance(row.get("correctness"), dict) and row.get("correctness", {}).get("field_correctness") is not None]
        samples = [sample for sample in (gpu_samples or []) if sample.get("config_id") == config_id and sample.get("phase") == "measured"]
        gpu_util = stats(sample["gpu_utilization_percent"] for sample in samples if sample.get("gpu_utilization_percent") is not None)
        vram = stats(sample["memory_used_mb"] for sample in samples if sample.get("memory_used_mb") is not None)
        result.append({"config_id": config_id, "document_type": document_type, "axis": rows[0].get("axis"), "value": rows[0].get("value"), "status": "ok" if all(row.get("status") == "ok" for row in rows) else "failed", "latency_seconds": latency["median"], "latency_iqr_seconds": latency["iqr"], "latency_min_seconds": latency["min"], "latency_max_seconds": latency["max"], "throughput_per_second": throughput["median"], "aggregate_throughput_per_second": aggregate_throughput["median"], "peak_vram_mb": vram["max"] if vram["max"] is not None else max((row.get("peak_vram_mb") for row in rows if row.get("peak_vram_mb") is not None), default=None), "median_vram_mb": vram["median"], "peak_gpu_utilization_percent": gpu_util["max"], "median_gpu_utilization_percent": gpu_util["median"], "semantic_change_count": max((row.get("semantic_change_count", 0) for row in rows), default=0), "confidence_only_change_count": max((row.get("confidence_only_change_count", 0) for row in rows), default=0), "field_correctness": sum(correctness) / len(correctness) if correctness else None, "failures": sum(int(row.get("failures", 0)) for row in rows), "cleanup_verified": all(row.get("cleanup_verified", False) for row in rows)})
    return result
