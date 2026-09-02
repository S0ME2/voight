"""Alternating fresh-server recognizer A/B for the finalized CPU profile."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.maintained.model_matrix_benchmark import (
    Server,
    _available_memory,
    _post,
    _score,
    _stage_totals,
)
from benchmarks.maintained.pipeline_breakdown import discover_dataset, validate_and_manifest

DOC_TYPES = ("passport", "id_card", "driving_license")
ORDER = ("PP-OCRv6_medium_rec", "latin_PP-OCRv5_mobile_rec") * 2
BASE_ENV = {
    "RUNTIME_TARGET": "cpu",
    "OCR_DEVICE": "cpu",
    "CPU_THREADS": "4",
    "TEXT_RECOGNITION_PROCESSES": "1",
    "LOCALIZATION_BATCH_SIZE": "4",
    "TEXT_DETECTION_BATCH_SIZE": "8",
    "TEXT_RECOGNITION_BATCH_SIZE": "4",
    "MRZ_RECOGNITION_BATCH_SIZE": "16",
    "TEXT_RECOGNITION_PACKING": "fixed-width",
    "TEXT_DETECTOR_MODEL": "PP-OCRv6_medium_det",
    "TEXT_RECOGNIZER_BACKEND": "paddle",
    "DOCALIGNER_MODEL": "fastvit_sa24",
    "DOCALIGNER_MODEL_TYPE": "heatmap",
    "MRZ_RECOGNIZER_BACKEND": "generic-paddle",
    "MRZ_RECOGNIZER_MODEL": "20250221",
    "OCR_MAX_SIDE": "3000",
    "OCR_CONTRAST": "1.25",
    "MRZ_POLYGON_PADDING_RATIO": "0.03",
    "DOCALIGNER_PADDING": "100",
    "DRIVING_LICENSE_MIN_OVERLAP_RATIO": "0.30",
}


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "models/benchmark")))
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/benchmarks/10.recognizer-a-b-comparison")
    parser.add_argument("--port", type=int, default=8015)
    parser.add_argument("--timeout", type=float, default=900)
    return parser.parse_args()


def _identities(payload: dict[str, Any]) -> dict[str, Any]:
    diagnostics = payload.get("diagnostics", {})
    line_filter = diagnostics.get("line_filter", {})

    def calls(stage: dict[str, Any]) -> list[dict[str, Any]]:
        keep = (
            "role", "submitted_batch_size", "tensor_batch_size", "tensor_batch_sizes",
            "input_widths", "input_heights", "submitted_input_shapes",
        )
        return [{key: call[key] for key in keep if key in call} for call in stage.get("calls", [])]

    localization = {
        name: calls(stage)
        for name, stage in diagnostics.get("localization", {}).items()
        if isinstance(stage, dict)
    }
    return {
        "sample_counts": diagnostics.get("sample_counts", {}),
        "sample_records": diagnostics.get("sample_records", []),
        "detected_line_count": line_filter.get("detected_line_count"),
        "recognition_candidate_count": line_filter.get("recognition_candidate_count"),
        "filtered_before_recognition_count": line_filter.get("filtered_before_recognition_count"),
        "line_counts_by_role": diagnostics.get("line_counts_by_role", {}),
        "mrz_crop_count": diagnostics.get("sample_counts", {}).get("mrz", 0),
        "localization_batches": localization,
        "detection_batches": calls(diagnostics.get("text_detection", {})),
        "recognition_batches": calls(diagnostics.get("text_recognition", {})),
        "mrz_batches": calls(diagnostics.get("mrz_recognition", {})),
    }


def _merge_scores(scores: list[dict[str, Any]]) -> dict[str, Any]:
    merged = {key: sum(score[key] for score in scores if isinstance(score.get(key), (int, float))) for key in scores[0]}
    for key, numerator, denominator in (
        ("field_correctness", "field_exact", "field_total"),
        ("document_correctness", "document_exact", "document_total"),
        ("mrz_found_rate", "mrz_found", "mrz_documents"),
        ("mrz_exact_match_rate", "mrz_full_exact", "mrz_documents"),
        ("mrz_line_accuracy", "mrz_line_exact", "mrz_lines"),
        ("mrz_character_accuracy", "mrz_characters", "mrz_character_total"),
    ):
        if key == "document_correctness":
            merged[key] = merged[numerator] / sum(score["document_total"] for score in scores)
        else:
            merged[key] = merged[numerator] / merged[denominator] if merged[denominator] else None
    return merged


def _document_total(score: dict[str, Any], count: int) -> dict[str, Any]:
    return {**score, "document_total": count}


def main() -> int:
    cli = args()
    documents, dataset = validate_and_manifest(cli.dataset_root)
    by_kind = {kind: [doc for doc in documents if doc.document_type == kind] for kind in DOC_TYPES}
    output = cli.output_dir / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output.mkdir(parents=True, exist_ok=False)
    (output / "raw").mkdir()
    (output / "server_logs").mkdir()
    manifest = {
        "dataset": dataset,
        "order": list(ORDER),
        "warmup_count_per_route": 1,
        "measured_runs": 4,
        "fresh_server_per_run": True,
        "cpu_only": True,
        "env": {**BASE_ENV, "MODEL_DIR": str(cli.model_dir.resolve())},
        "benchmark_script": str(Path(__file__).resolve()),
        "git_commit_at_start": os.popen("git rev-parse HEAD").read().strip(),
        "git_dirty_at_start": bool(os.popen("git status --porcelain").read().strip()),
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    rows = []
    workload = None

    for index, recognizer in enumerate(ORDER, 1):
        name = f"{index:02d}_{recognizer}"
        run_dir = output / name
        run_dir.mkdir()
        env = {**BASE_ENV, "TEXT_RECOGNIZER_MODEL": recognizer, "MODEL_DIR": str(cli.model_dir.resolve())}
        server = Server(cli, run_dir, env)
        lifecycle = {"memory_before_mb": (_available_memory() or 0) / 1024 / 1024}
        payloads: dict[str, dict[str, Any]] = {}
        client_seconds = {}
        scores = []
        started = time.perf_counter()
        try:
            ready = server.start()
            (run_dir / "loaded_configuration.json").write_text(json.dumps(ready, indent=2), encoding="utf-8")
            warmup_dir = run_dir / "warmup"
            warmup_dir.mkdir()
            for kind in DOC_TYPES:
                payload, seconds = _post(kind, by_kind[kind], cli.port, cli.timeout)
                (warmup_dir / f"{kind}.json").write_text(json.dumps({"client_wall_seconds": seconds, "response": payload}, indent=2), encoding="utf-8")
            run_dir.joinpath("raw").mkdir()
            for kind in DOC_TYPES:
                payload, seconds = _post(kind, by_kind[kind], cli.port, cli.timeout)
                payloads[kind] = payload
                client_seconds[kind] = seconds
                scores.append(_document_total(_score(kind, by_kind[kind], payload), len(by_kind[kind])))
                (run_dir / "raw" / f"{kind}.json").write_text(json.dumps({"client_wall_seconds": seconds, "response": payload}, indent=2), encoding="utf-8")
            current_workload = {kind: _identities(payload) for kind, payload in payloads.items()}
            if workload is None:
                workload = current_workload
                (output / "workload_reference.json").write_text(json.dumps(workload, indent=2), encoding="utf-8")
            elif current_workload != workload:
                raise RuntimeError(f"workload mismatch in {name}; see raw responses before accepting timings")
            server._sample()
            elapsed = time.perf_counter() - started
            stages = _stage_totals(list(payloads.values()))
            total_seconds = sum(float(payload.get("total_seconds", 0.0)) for payload in payloads.values())
            row = {
                "run": index,
                "recognizer": recognizer,
                "status": "ok",
                "total_seconds": total_seconds,
                "client_seconds": sum(client_seconds.values()),
                "wall_seconds": elapsed,
                "throughput_docs_per_second": len(documents) / total_seconds if total_seconds else None,
                "stages": stages,
                "other_seconds": max(0.0, total_seconds - sum(stages.values())),
                "scores": _merge_scores(scores),
                "request_count": len(payloads),
                "success_count": sum(payload.get("succeeded", 0) for payload in payloads.values()),
                "failure_count": sum(payload.get("failed", 0) for payload in payloads.values()),
                "workload_equivalent_to_reference": True,
            }
            (run_dir / "measurement.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            rows.append(row)
        except Exception as error:
            (run_dir / "failure.json").write_text(json.dumps({"error": f"{type(error).__name__}: {error}"}, indent=2), encoding="utf-8")
            raise
        finally:
            lifecycle.update(server.stop())
            memory_after = _available_memory()
            lifecycle["memory_after_shutdown_mb"] = memory_after / 1024 / 1024 if memory_after else None
            (run_dir / "lifecycle.json").write_text(json.dumps(lifecycle, indent=2), encoding="utf-8")
        if not lifecycle.get("cleanup_verified"):
            raise RuntimeError(f"server cleanup failed for {name}: {lifecycle}")

    summary = {
        "runs": rows,
        "median_by_recognizer": {
            recognizer: {
                "total_seconds": statistics.median(row["total_seconds"] for row in rows if row["recognizer"] == recognizer),
                "stages": {stage: statistics.median(row["stages"][stage] for row in rows if row["recognizer"] == recognizer) for stage in rows[0]["stages"]},
            }
            for recognizer in sorted(set(ORDER))
        },
        "workload_equivalent": True,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output / "workload_equivalence.json").write_text(json.dumps({"equivalent": True, "reference": workload}, indent=2), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
