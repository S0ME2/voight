"""Compare saved CPU and GPU benchmark artifacts without starting anything."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from benchmarks.gpu.helpers import csv_write


def _rows(directory: Path) -> list[dict]:
    for name in ("comparison.csv", "raw_results.csv", "raw_measurements.csv", "results.csv"):
        path = directory / name
        if path.is_file():
            with path.open(newline="", encoding="utf-8") as handle:
                return list(csv.DictReader(handle))
    raise FileNotFoundError(f"no supported benchmark CSV in {directory}")


def _number(row: dict, *keys: str) -> float | None:
    for key in keys:
        try:
            value = row.get(key)
            if value not in (None, "", "None"):
                return float(value)
        except (TypeError, ValueError):
            pass
    return None


def compare(gpu_dir: Path, cpu_dir: Path, output: Path) -> list[dict]:
    gpu, cpu = _rows(gpu_dir), _rows(cpu_dir)
    cpu_baseline = cpu[0] if cpu else {}
    cpu_by_type = {row.get("document_type"): row for row in cpu if row.get("document_type")}
    rows = []
    for row in gpu or [{}]:
        cpu_row = cpu_by_type.get(row.get("document_type"), cpu_baseline)
        cpu_latency = _number(cpu_row, "latency_seconds", "median_latency_seconds", "median_e2e_seconds", "e2e_seconds", "total_latency_seconds")
        cpu_throughput = _number(cpu_row, "throughput_per_second", "throughput_docs_per_second", "docs_per_second", "items_per_second")
        latency = _number(row, "latency_seconds", "median_latency_seconds", "median_e2e_seconds", "e2e_seconds", "total_latency_seconds")
        throughput = _number(row, "aggregate_throughput_per_second", "throughput_per_second", "throughput_docs_per_second", "docs_per_second", "items_per_second")
        rows.append({"cpu_artifact": str(cpu_dir), "gpu_artifact": str(gpu_dir), "configuration": row.get("config_id", "baseline"), "cpu_latency_seconds": cpu_latency, "gpu_latency_seconds": latency, "speedup": cpu_latency / latency if cpu_latency and latency else None, "cpu_throughput_per_second": cpu_throughput, "gpu_throughput_per_second": throughput, "throughput_speedup": throughput / cpu_throughput if cpu_throughput and throughput else None, "cpu_rss_mb": _number(cpu_row, "peak_rss_mb", "peak_rss_mb_max", "peak_process_memory_mb", "rss_mb"), "gpu_host_rss_mb": _number(row, "host_rss_mb", "peak_rss_mb"), "gpu_vram_mb": _number(row, "peak_vram_mb"), "semantic_change_count": _number(row, "semantic_change_count"), "correctness": _number(row, "correctness", "field_correctness", "document_correctness")})
        rows[-1]["document_type"] = row.get("document_type")
        rows[-1]["confidence_only_change_count"] = _number(row, "confidence_only_change_count")
    output.mkdir(parents=True, exist_ok=True)
    csv_write(output / "cpu_gpu_comparison.csv", rows)
    (output / "cpu_gpu_comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    points = []
    for index, row in enumerate(rows):
        if row.get("gpu_latency_seconds") is not None and row.get("gpu_throughput_per_second") is not None:
            points.append(f'<circle cx="{50 + index * 80}" cy="{250 - min(220, float(row["gpu_throughput_per_second"]) * 20)}" r="4" fill="#1769aa"/><text x="{45 + index * 80}" y="280" font-size="10">{row["configuration"]}</text>')
    (output / "pareto.svg").write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="900" height="300"><rect width="100%" height="100%" fill="white"/><text x="20" y="20">GPU throughput by latency configuration</text>{"".join(points)}</svg>\n', encoding="utf-8")
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gpu_output", type=Path)
    parser.add_argument("cpu_output", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/benchmarks/cpu_gpu_comparison"))
    args = parser.parse_args(argv)
    print(json.dumps(compare(args.gpu_output, args.cpu_output, args.output), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
