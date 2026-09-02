"""Reproducible, CPU-only analysis of the optimization-six benchmark artifacts.

This script only reads benchmark evidence and annotations.  All derived files
are written below ``09.visual-analysis``; it never imports the OCR pipeline or
changes production settings.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DOC_TYPES = ("passport", "id_card", "driving_license")
DISPLAY_TYPES = {"passport": "Passport", "id_card": "ID card", "driving_license": "Driving licence"}
COLORS = {"Baseline": "#4c566a", "SAFE_OBSERVED": "#2e8b57", "FASTEST_EXPERIMENTAL": "#d95f02", "medium": "#2e8b57", "tiny": "#d95f02", "small": "#7570b3"}
STAGES = ("localization", "detection", "recognition", "mrz_recognition", "other")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def median(values: Iterable[float]) -> float | None:
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return statistics.median(values) if values else None


def num(value: Any) -> float | None:
    try:
        value = float(value)
        return value if math.isfinite(value) else None
    except (TypeError, ValueError):
        return None


def lev(left: str, right: str) -> int:
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        next_row = [i]
        for j, b in enumerate(right, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = next_row
    return row[-1]


def wilson(correct: int, total: int, z: float = 1.96) -> tuple[float | None, float | None]:
    if not total:
        return None, None
    p = correct / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt((p * (1 - p) + z * z / (4 * total)) / total) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)


def json_or_empty(path: Path) -> Any:
    try:
        return read_json(path)
    except (OSError, json.JSONDecodeError):
        return None


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "(no measured rows)"
    headers = [str(c) for c in frame.columns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = ["" if v is None or (isinstance(v, (float, np.floating)) and math.isnan(float(v))) else str(v) for v in row]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


class Analysis:
    def __init__(self, root: Path, annotations_root: Path, output: Path) -> None:
        self.root = root
        self.annotations_root = annotations_root
        self.output = output
        self.plots = output / "plots"
        self.tables = output / "tables"
        self.observed = output / "observed_tables"
        self.data = output / "data"
        for directory in (self.output, self.plots, self.tables, self.observed, self.data):
            directory.mkdir(parents=True, exist_ok=True)
        self.plot_index: list[dict[str, str]] = []
        self.annotations = self.load_annotations()
        self.inventory = self.build_inventory()
        self.all_jsonl = self.load_all_jsonl()

    def load_annotations(self) -> dict[str, dict[str, Any]]:
        result = {}
        for path in sorted(self.annotations_root.glob("*/*.json")):
            value = json_or_empty(path)
            if isinstance(value, dict) and value.get("id"):
                result[value["id"]] = value
        return result

    def build_inventory(self) -> pd.DataFrame:
        rows = []
        for path in sorted(self.root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {".json", ".jsonl", ".csv"}:
                continue
            try:
                size = path.stat().st_size
            except OSError:
                size = None
            rows.append({"relative_path": str(path.relative_to(self.root)), "format": path.suffix.lower().lstrip("."), "bytes": size})
        frame = pd.DataFrame(rows)
        frame.to_csv(self.data / "artifact_inventory.csv", index=False)
        return frame

    def load_all_jsonl(self) -> dict[str, list[dict[str, Any]]]:
        result = {}
        for path in sorted(self.root.rglob("*.jsonl")):
            try:
                result[str(path.relative_to(self.root))] = read_jsonl(path)
            except (OSError, json.JSONDecodeError):
                result[str(path.relative_to(self.root))] = []
        return result

    @staticmethod
    def observed_row(row: dict[str, Any], source_file: str, source_row: int, fields: tuple[str, ...], include_stages: bool = False) -> dict[str, Any]:
        result = {"source_file": source_file, "source_row": source_row}
        for field in fields:
            result[field] = row.get(field)
        if include_stages:
            for stage, value in (row.get("stages") or {}).items():
                result[f"stage_{stage}_seconds"] = value
        return result

    def write_observed_table(self, name: str, rows: list[dict[str, Any]], description: str) -> None:
        frame = pd.DataFrame(rows)
        csv_path = self.observed / f"{name}.csv"
        md_path = self.observed / f"{name}.md"
        frame.to_csv(csv_path, index=False)
        md_path.write_text(f"# {name}\n\n{description}\n\n{markdown_table(frame)}\n", encoding="utf-8")

    def write_observed_tables(self) -> None:
        """Export source-emitted measurement fields without calculating new metrics."""
        raw_specs = {
            "01.baseline": [
                ("passport", "01.baseline/02.passport-full/raw_measurements.jsonl"),
                ("id_card", "01.baseline/03.id-card-full/raw_measurements.jsonl"),
                ("driving_license", "01.baseline/04.driving-license-full/raw_measurements.jsonl"),
                ("passport_visible_probe", "01.baseline/01.passport-visible-probe/raw_measurements.jsonl"),
            ],
            "02.recognition-batch": [("", "02.recognition-batch/raw.jsonl")],
            "03.split-visible-mrz": [("", "03.split-visible-mrz/raw.jsonl")],
            "08.final-result": [("", "08.final-result/raw.jsonl")],
        }
        timing_fields = ("variant", "document_type", "repeat", "logical_count", "physical_count", "status", "total_seconds", "client_seconds", "server_seconds", "error")
        for name, specs in raw_specs.items():
            rows = []
            for _, relative in specs:
                for index, row in enumerate(self.all_jsonl.get(relative, []), 1):
                    rows.append(self.observed_row(row, relative, index, timing_fields, include_stages=True))
            self.write_observed_table(name, rows, "Direct fields copied from the experiment JSONL. Stage columns are the emitted stage timings; blank means the source row did not contain that field. No throughput, speedup, or accuracy is calculated here.")

        model_fields = ("model", "configuration", "batch_size", "repeat", "status", "total_recognition_seconds", "lines", "lines_per_second", "milliseconds_per_line", "model_call_count", "submitted_batch_sizes", "tensor_batch_sizes", "exact_text_match_rate", "differing_lines", "score_difference_count", "error")
        rows = [self.observed_row(row, "04.recognizer-models/raw.jsonl", index, model_fields) for index, row in enumerate(self.all_jsonl.get("04.recognizer-models/raw.jsonl", []), 1)]
        self.write_observed_table("04.recognizer-models", rows, "Direct fixed-corpus recognizer benchmark fields. `lines_per_second` and `milliseconds_per_line` are present only when emitted by the model benchmark.")

        runtime_fields = ("backend", "threads", "status", "load_seconds", "median_seconds", "lines_per_second", "error")
        rows = [self.observed_row(row, "05.cpu-runtime/raw.jsonl", index, runtime_fields) for index, row in enumerate(self.all_jsonl.get("05.cpu-runtime/raw.jsonl", []), 1)]
        self.write_observed_table("05.cpu-runtime", rows, "Direct CPU runtime rows. The source stores median inference time, not individual repeat samples.")

        row_fields = ("kind", "document_id", "variant", "repeat", "seconds", "line_count", "lines_per_second", "correctness")
        rows = []
        for index, row in enumerate(self.all_jsonl.get("06.mrz-rows/raw.jsonl", []), 1):
            value = self.observed_row(row, "06.mrz-rows/raw.jsonl", index, row_fields)
            value["correctness"] = json.dumps(value["correctness"], sort_keys=True) if value.get("correctness") is not None else None
            for stage, stage_value in (row.get("stages") or {}).items(): value[f"stage_{stage}_seconds"] = stage_value
            rows.append(value)
        self.write_observed_table("06.mrz-rows", rows, "Direct MRZ row experiment fields. `lines_per_second` is retained only where the source emitted it; no seconds-per-line inverse is added.")

        fallback_fields = ("kind", "document_id", "fast_seconds", "valid", "false_accept", "fallback", "fallback_seconds", "total_seconds")
        rows = [self.observed_row(row, "07.fast-fallback/raw.jsonl", index, fallback_fields) for index, row in enumerate(self.all_jsonl.get("07.fast-fallback/raw.jsonl", []), 1) if row.get("document_id")]
        self.write_observed_table("07.fast-fallback", rows, "Direct per-document fast/fallback fields. MRZ output strings are intentionally omitted; validity and false-accept flags are copied as emitted.")

        summary = json_or_empty(self.root / "07.fast-fallback/summary.json") or {}
        summary_rows = [{"source_file": "07.fast-fallback/summary.json", "metric": key, "value": value} for key, value in summary.items() if not isinstance(value, (dict, list))]
        self.write_observed_table("07.fast-fallback_summary", summary_rows, "Scalar fields copied directly from the experiment summary JSON; no rates are recalculated.")

        correctness_rows = []
        correctness_paths = [
            path for path in sorted(self.root.rglob("correctness.json"))
        ]
        for path in correctness_paths:
            payload = json_or_empty(path) or {}
            stack = [("", payload)]
            while stack:
                metric_path, value = stack.pop()
                if isinstance(value, dict):
                    for key, child in value.items(): stack.append((f"{metric_path}.{key}".strip("."), child))
                elif isinstance(value, (int, float, str, bool)) or value is None:
                    correctness_rows.append({"source_file": str(path.relative_to(self.root)), "metric_path": metric_path, "value": value})
        for relative in ("02.recognition-batch/summary.json", "03.split-visible-mrz/summary.json"):
            payload = json_or_empty(self.root / relative) or {}
            for key, value in (payload.get("correctness") or {}).items():
                stack = [(key, value)]
                while stack:
                    metric_path, child = stack.pop()
                    if isinstance(child, dict):
                        for subkey, subvalue in child.items(): stack.append((f"{metric_path}.{subkey}", subvalue))
                    elif isinstance(child, (int, float, str, bool)) or child is None:
                        correctness_rows.append({"source_file": relative, "metric_path": metric_path, "value": child})
        self.write_observed_table("correctness_metrics", correctness_rows, "Scalar correctness fields copied from benchmark-emitted correctness JSON and summary correctness sections. These are source metrics, not re-scored values.")

        index = ["# Observed experiment tables", "", "These tables contain only fields emitted by the benchmark artifacts. No speedups, inverses, joins, percentages, confidence intervals, per-field rescoring, or other derived metrics are added. Blank cells mean the source did not emit a value.", "", "Important: `milliseconds_per_line` and `lines_per_second` are directly emitted by Experiment 3; Experiment 5 emits `lines_per_second` only for some row-split rows. There is intentionally no calculated `seconds_per_line` column.", ""]
        for path in sorted(self.observed.glob("*.csv")):
            index.append(f"- [{path.name}]({path.name}) · [{path.with_suffix('.md').name}]({path.with_suffix('.md').name})")
        (self.observed / "INDEX.md").write_text("\n".join(index) + "\n", encoding="utf-8")

    def annotations_coverage(self) -> pd.DataFrame:
        rows = []
        for kind in DOC_TYPES:
            docs = [a for a in self.annotations.values() if a.get("document_type") == kind]
            counts = Counter()
            for doc in docs:
                for field in doc.get("fields", {}).values():
                    counts[field.get("state", "unannotated")] += 1
            total = sum(counts.values())
            rows.append({
                "document_type": kind,
                "documents": len(docs),
                "total_fields": total,
                "scorable_fields": counts.get("value", 0) + counts.get("empty", 0),
                "value_fields": counts.get("value", 0),
                "empty_fields": counts.get("empty", 0),
                "unreadable_fields": counts.get("unreadable", 0),
                "unannotated_fields": total - sum(counts.get(s, 0) for s in ("value", "empty", "unreadable")),
                "mrz_documents": sum(bool(d.get("mrz", {}).get("lines")) for d in docs),
                "mrz_lines": sum(len(d.get("mrz", {}).get("lines", [])) for d in docs),
            })
        frame = pd.DataFrame(rows)
        frame.to_csv(self.tables / "annotation_coverage.csv", index=False)
        return frame

    def truth_fields(self, document_id: str) -> dict[str, dict[str, Any]]:
        return self.annotations.get(document_id, {}).get("fields", {})

    def scorable(self, entry: dict[str, Any]) -> bool:
        return entry.get("state") in {"value", "empty"}

    def output_rows(self) -> dict[tuple[str, str], dict[str, Any]]:
        rows = self.all_jsonl.get("08.final-result/raw.jsonl", [])
        result = {}
        for row in rows:
            if row.get("repeat") == 1 and row.get("outputs") and row.get("variant") in {"SAFE_OBSERVED", "FASTEST_EXPERIMENTAL"}:
                result[(row.get("variant"), row.get("document_type"))] = row
        return result

    def score_outputs(self, variant: str, kind: str, outputs: dict[str, Any], modality: str = "visible") -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, Counter]:
        aggregate = {"scorable_count": 0, "exact_correct": 0, "character_correct": 0, "character_total": 0, "missing": 0, "incorrect": 0}
        field_rows, doc_rows = [], []
        confusions = Counter()
        for document_id, output in outputs.items():
            annotation = self.annotations.get(document_id, {})
            if modality == "visible":
                doc_total = doc_correct = doc_chars = doc_char_total = 0
                for field, entry in annotation.get("fields", {}).items():
                    if not self.scorable(entry):
                        continue
                    expected = "" if entry.get("state") == "empty" else str(entry.get("value") or "")
                    actual = output.get("fields", {}).get(field)
                    actual = "" if actual is None else str(actual)
                    distance = lev(expected, actual)
                    exact = actual == expected
                    chars = max(0, len(expected) - distance)
                    aggregate["scorable_count"] += 1
                    aggregate["exact_correct"] += int(exact)
                    aggregate["character_correct"] += chars
                    aggregate["character_total"] += max(len(expected), 1)
                    aggregate["missing"] += int(not actual)
                    aggregate["incorrect"] += int(bool(actual) and not exact)
                    doc_total += 1; doc_correct += int(exact); doc_chars += chars; doc_char_total += max(len(expected), 1)
                    for left, right in zip(expected, actual):
                        if left != right:
                            confusions[(left, right)] += 1
                    if len(actual) < len(expected):
                        confusions[("<", "omission")] += len(expected) - len(actual)
                    elif len(actual) > len(expected):
                        confusions[("insertion", ">") ] += len(actual) - len(expected)
                    field_rows.append({"document_type": kind, "field": field, "configuration": variant, "n": 1, "exact_correct": int(exact), "exact_rate": int(exact), "character_accuracy": chars / max(len(expected), 1), "state": entry.get("state")})
                doc_rows.append({"document_type": kind, "document_id": document_id, "configuration": variant, "scorable_fields": doc_total, "correct_fields": doc_correct, "accuracy": doc_correct / doc_total if doc_total else None, "character_accuracy": doc_chars / doc_char_total if doc_char_total else None, "regressions_vs_baseline": None, "improvements_vs_baseline": None})
            else:
                expected_lines = [line for line in annotation.get("mrz", {}).get("lines", []) if isinstance(line, str)]
                if not expected_lines:
                    continue
                actual_lines = [str(line) for line in output.get("mrz", [])]
                full = len(expected_lines) == len(actual_lines) and all(a == b for a, b in zip(actual_lines, expected_lines))
                line_exact = 0; chars = 0; total_chars = 0
                for index, expected in enumerate(expected_lines):
                    actual = actual_lines[index] if index < len(actual_lines) else ""
                    distance = lev(expected, actual); line_exact += int(actual == expected); chars += max(0, len(expected) - distance); total_chars += len(expected)
                    for left, right in zip(expected, actual):
                        if left != right: confusions[(left, right)] += 1
                aggregate.setdefault("documents", 0); aggregate.setdefault("full_exact", 0); aggregate.setdefault("lines", 0); aggregate.setdefault("line_exact", 0)
                aggregate["documents"] += 1; aggregate["full_exact"] += int(full); aggregate["lines"] += len(expected_lines); aggregate["line_exact"] += line_exact
                aggregate["character_correct"] += chars; aggregate["character_total"] += total_chars
                doc_rows.append({"document_type": kind, "document_id": document_id, "configuration": variant, "scorable_fields": len(expected_lines), "correct_fields": line_exact, "accuracy": line_exact / len(expected_lines) if expected_lines else None, "full_exact": int(full), "character_accuracy": chars / total_chars if total_chars else None, "regressions_vs_baseline": None, "improvements_vs_baseline": None})
        if modality == "visible":
            aggregate["exact_rate"] = aggregate["exact_correct"] / aggregate["scorable_count"] if aggregate["scorable_count"] else None
            aggregate["character_accuracy"] = aggregate["character_correct"] / aggregate["character_total"] if aggregate["character_total"] else None
        else:
            aggregate["exact_rate"] = aggregate.get("full_exact", 0) / aggregate.get("documents", 0) if aggregate.get("documents") else None
            aggregate["line_exact_rate"] = aggregate.get("line_exact", 0) / aggregate.get("lines", 0) if aggregate.get("lines") else None
            aggregate["character_accuracy"] = aggregate["character_correct"] / aggregate["character_total"] if aggregate["character_total"] else None
        fields = pd.DataFrame(field_rows)
        if not fields.empty:
            fields = fields.groupby(["document_type", "field", "configuration", "state"], as_index=False).agg(n=("n", "sum"), exact_correct=("exact_correct", "sum"), exact_rate=("exact_rate", "mean"), character_accuracy=("character_accuracy", "mean"))
        return aggregate, fields, pd.DataFrame(doc_rows), confusions

    def paired_transitions(self, kind: str, left_variant: str, right_variant: str, left_outputs: dict[str, Any], right_outputs: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
        counts = Counter(); rows = []
        for document_id in sorted(set(left_outputs) & set(right_outputs)):
            for field, entry in self.truth_fields(document_id).items():
                if not self.scorable(entry):
                    continue
                expected = "" if entry.get("state") == "empty" else str(entry.get("value") or "")
                left = str(left_outputs[document_id].get("fields", {}).get(field) or "")
                right = str(right_outputs[document_id].get("fields", {}).get(field) or "")
                lc, rc = left == expected, right == expected
                category = "BOTH_CORRECT" if lc and rc else "REGRESSION" if lc else "IMPROVEMENT" if rc else "BOTH_WRONG"
                counts[category] += 1
                rows.append({"document_type": kind, "document_id": document_id, "field": field, "baseline_configuration": left_variant, "configuration": right_variant, "transition": category})
        aggregate = pd.DataFrame([{ "document_type": kind, "configuration": right_variant, "both_correct": counts["BOTH_CORRECT"], "regressions": counts["REGRESSION"], "improvements": counts["IMPROVEMENT"], "both_wrong": counts["BOTH_WRONG"], "evaluable": sum(counts.values()), "net_change": counts["IMPROVEMENT"] - counts["REGRESSION"]}])
        return aggregate, pd.DataFrame(rows)

    def paired_mrz_transitions(self, kind: str, left_outputs: dict[str, Any], right_outputs: dict[str, Any]) -> pd.DataFrame:
        counts = Counter()
        for document_id in sorted(set(left_outputs) & set(right_outputs)):
            expected_lines = self.annotations.get(document_id, {}).get("mrz", {}).get("lines", [])
            left_lines = left_outputs[document_id].get("mrz", [])
            right_lines = right_outputs[document_id].get("mrz", [])
            for index, expected in enumerate(expected_lines):
                left = left_lines[index] if index < len(left_lines) else ""
                right = right_lines[index] if index < len(right_lines) else ""
                lc, rc = left == expected, right == expected
                counts["BOTH_CORRECT" if lc and rc else "REGRESSION" if lc else "IMPROVEMENT" if rc else "BOTH_WRONG"] += 1
        return pd.DataFrame([{ "document_type": kind, "configuration": "FASTEST_EXPERIMENTAL (tiny)", "modality": "MRZ", "both_correct": counts["BOTH_CORRECT"], "regressions": counts["REGRESSION"], "improvements": counts["IMPROVEMENT"], "both_wrong": counts["BOTH_WRONG"], "evaluable": sum(counts.values()), "net_change": counts["IMPROVEMENT"] - counts["REGRESSION"]}])

    def timing_rows(self) -> pd.DataFrame:
        rows = []
        # Original full-pipeline baseline.
        for kind in DOC_TYPES:
            path = {"passport": "01.baseline/02.passport-full/raw_measurements.jsonl", "id_card": "01.baseline/03.id-card-full/raw_measurements.jsonl", "driving_license": "01.baseline/04.driving-license-full/raw_measurements.jsonl"}[kind]
            for row in self.all_jsonl.get(path, []):
                if row.get("status") == "ok" and int(row.get("logical_count", 0)) == len([a for a in self.annotations.values() if a.get("document_type") == kind]):
                    rows.append({"experiment": "01.baseline", "document_type": kind, "configuration": "Baseline", "repeat": row.get("repeat"), "median_seconds": row.get("total_seconds"), "seconds": row.get("total_seconds"), "docs_per_second": len([a for a in self.annotations.values() if a.get("document_type") == kind]) / float(row.get("total_seconds")), "lines_per_second": None, "stages": row.get("stages", {})})
        # Final integrated runs.
        for row in self.all_jsonl.get("08.final-result/raw.jsonl", []):
            if row.get("document_type") in DOC_TYPES:
                n = len([a for a in self.annotations.values() if a.get("document_type") == row["document_type"]])
                rows.append({"experiment": "08.final-result", "document_type": row["document_type"], "configuration": row.get("variant"), "repeat": row.get("repeat"), "median_seconds": row.get("seconds"), "seconds": row.get("seconds"), "docs_per_second": n / float(row["seconds"]), "lines_per_second": None, "stages": row.get("stages", {})})
        # Experiment 1 visible timing.
        for row in self.all_jsonl.get("02.recognition-batch/raw.jsonl", []):
            batch = row.get("variant", "").split("batch=")[-1] if "batch=" in row.get("variant", "") else None
            if batch is not None and row.get("document_type") in DOC_TYPES:
                n = len([a for a in self.annotations.values() if a.get("document_type") == row["document_type"]])
                rows.append({"experiment": "02.recognition-batch", "document_type": row["document_type"], "configuration": f"batch={batch}", "repeat": row.get("repeat"), "median_seconds": row.get("total_seconds"), "seconds": row.get("total_seconds"), "docs_per_second": n / float(row["total_seconds"]), "lines_per_second": None, "stages": row.get("stages", {})})
        # Experiment 2 modes.
        for row in self.all_jsonl.get("03.split-visible-mrz/raw.jsonl", []):
            if row.get("variant") in {"CURRENT_COMBINED", "SPLIT_SAME_BATCH", "SPLIT_TUNED_BATCH"}:
                n = len([a for a in self.annotations.values() if a.get("document_type") == row["document_type"]])
                rows.append({"experiment": "03.split-visible-mrz", "document_type": row["document_type"], "configuration": row["variant"], "repeat": row.get("repeat"), "median_seconds": row.get("total_seconds"), "seconds": row.get("total_seconds"), "docs_per_second": n / float(row["total_seconds"]), "lines_per_second": None, "stages": row.get("stages", {})})
        # Fixed-corpus recognizer benchmark.
        for row in self.all_jsonl.get("04.recognizer-models/raw.jsonl", []):
            if row.get("status") == "ok":
                rows.append({"experiment": "04.recognizer-models", "document_type": "fixed_corpus", "configuration": row.get("model"), "repeat": row.get("repeat"), "median_seconds": row.get("total_recognition_seconds"), "seconds": row.get("total_recognition_seconds"), "docs_per_second": None, "lines_per_second": row.get("lines_per_second"), "milliseconds_per_line": row.get("milliseconds_per_line"), "exact_text_match_rate": row.get("exact_text_match_rate"), "stages": {}})
        # Experiment 6 timing is per-document aggregate for the two visible modes.
        summary = json_or_empty(self.root / "07.fast-fallback/summary.json") or {}
        for item in summary.get("visible_modes", []):
            rows.append({"experiment": "07.fast-fallback", "document_type": item.get("kind"), "configuration": item.get("variant"), "repeat": 1, "median_seconds": item.get("seconds"), "seconds": item.get("seconds"), "docs_per_second": len([a for a in self.annotations.values() if a.get("document_type") == item.get("kind")]) / float(item.get("seconds")), "lines_per_second": None, "stages": {}})
        return pd.DataFrame(rows)

    def baseline_correctness(self) -> pd.DataFrame:
        rows = []
        paths = {"passport": "01.baseline/02.passport-full/correctness.json", "id_card": "01.baseline/03.id-card-full/correctness.json", "driving_license": "01.baseline/04.driving-license-full/correctness.json"}
        for kind, path in paths.items():
            value = json_or_empty(self.root / path) or {}
            candidates = [(k, v) for k, v in value.items() if k.startswith({"passport": "passport_full", "id_card": "id_card_full", "driving_license": "driving_license_full"}[kind]) and v.get("visible", {}).get("evaluated") == sum(1 for a in self.annotations.values() if a.get("document_type") == kind for e in a.get("fields", {}).values() if self.scorable(e))]
            if candidates:
                metric = candidates[0][1]
                vis = metric["visible"]; mrz = metric["mrz"]
                rows.append({"document_type": kind, "configuration": "Baseline", "scorable_count": vis.get("evaluated"), "exact_correct": vis.get("exact"), "exact_rate": vis.get("exact", 0) / vis.get("evaluated", 1), "character_accuracy": vis.get("characters", 0) / vis.get("character_total", 1), "MRZ_full_exact": mrz.get("full_exact"), "MRZ_documents": mrz.get("documents"), "MRZ_line_exact": mrz.get("line_exact"), "MRZ_lines": mrz.get("lines"), "parser_success": mrz.get("parser_success"), "validation_success": mrz.get("validation_success"), "false_valid_MRZ": None, "source": path})
        # Final outputs are scored from the actual stored candidate text.
        for (variant, kind), row in self.output_rows().items():
            vis, _, _, _ = self.score_outputs(variant, kind, row["outputs"], "visible")
            mrz, _, _, _ = self.score_outputs(variant, kind, row["outputs"], "mrz")
            rows.append({"document_type": kind, "configuration": variant, "scorable_count": vis.get("scorable_count"), "exact_correct": vis.get("exact_correct"), "exact_rate": vis.get("exact_rate"), "character_accuracy": vis.get("character_accuracy"), "MRZ_full_exact": mrz.get("full_exact"), "MRZ_documents": mrz.get("documents"), "MRZ_line_exact": mrz.get("line_exact"), "MRZ_lines": mrz.get("lines"), "parser_success": None, "validation_success": None, "false_valid_MRZ": None, "source": "08.final-result/raw.jsonl scored against annotations"})
        # V0/V1 aggregate was emitted by experiment 6 and is a distinct path.
        summary = json_or_empty(self.root / "07.fast-fallback/summary.json") or {}
        for mode in summary.get("visible_modes", []):
            correctness = next(iter(mode.get("correctness", {}).values()), {})
            vis = correctness.get("visible", {})
            mrz = correctness.get("mrz", {})
            false_accepts = sum(1 for raw in self.all_jsonl.get("07.fast-fallback/raw.jsonl", []) if raw.get("kind") == mode.get("kind") and raw.get("false_accept"))
            rows.append({"document_type": mode.get("kind"), "configuration": mode.get("variant"), "scorable_count": vis.get("evaluated"), "exact_correct": vis.get("exact"), "exact_rate": vis.get("exact", 0) / vis.get("evaluated", 1) if vis.get("evaluated") else None, "character_accuracy": vis.get("characters", 0) / vis.get("character_total", 1) if vis.get("character_total") else None, "MRZ_full_exact": mrz.get("full_exact"), "MRZ_documents": mrz.get("documents"), "MRZ_line_exact": mrz.get("line_exact"), "MRZ_lines": mrz.get("lines"), "parser_success": mrz.get("parser_success"), "validation_success": mrz.get("validation_success"), "false_valid_MRZ": false_accepts if mode.get("variant") == "V1_FAST_ONLY" else None, "source": "07.fast-fallback/summary.json"})
        frame = pd.DataFrame(rows).drop_duplicates(subset=["document_type", "configuration", "source"])
        for col in ("exact_rate", "character_accuracy"):
            frame[f"{col}_low"] = [wilson(int(r["exact_correct"] or 0), int(r["scorable_count"] or 0))[0] if col == "exact_rate" else None for _, r in frame.iterrows()]
            frame[f"{col}_high"] = [wilson(int(r["exact_correct"] or 0), int(r["scorable_count"] or 0))[1] if col == "exact_rate" else None for _, r in frame.iterrows()]
        frame.to_csv(self.tables / "accuracy_summary.csv", index=False)
        return frame

    def make_plot(self, name: str, title: str, description: str, caveat: str, draw) -> None:
        fig, ax = plt.subplots(figsize=(10, 6))
        draw(fig, ax)
        fig.suptitle(title, fontsize=14, y=0.99)
        fig.tight_layout()
        fig.savefig(self.plots / f"{name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(self.plots / f"{name}.svg", bbox_inches="tight")
        plt.close(fig)
        self.plot_index.append({"filename": f"{name}.png", "what": description, "interpretation": description, "caveat": caveat})

    def grouped_bars(self, ax, frame: pd.DataFrame, x: str, y: str, hue: str, order: list[str] | None = None, annotate: bool = True) -> None:
        if frame.empty:
            ax.text(0.5, 0.5, "No measured data", ha="center", va="center")
            return
        pivot = frame.pivot_table(index=x, columns=hue, values=y, aggfunc="median")
        if order:
            pivot = pivot.reindex(order)
        width = 0.8 / max(len(pivot.columns), 1)
        positions = np.arange(len(pivot.index))
        for index, column in enumerate(pivot.columns):
            values = pivot[column].to_numpy(dtype=float)
            bars = ax.bar(positions + (index - (len(pivot.columns) - 1) / 2) * width, values, width, label=str(column), color=COLORS.get(str(column), None))
            if annotate:
                for bar, value in zip(bars, values):
                    if np.isfinite(value):
                        ax.text(bar.get_x() + bar.get_width() / 2, value, f"{value:.3g}", ha="center", va="bottom", fontsize=8, rotation=90)
        ax.set_xticks(positions, [DISPLAY_TYPES.get(str(v), str(v)) for v in pivot.index])
        ax.legend(fontsize=8)

    def plot_performance(self, timings: pd.DataFrame) -> None:
        finals = timings[timings["experiment"].isin(["01.baseline", "08.final-result"])].copy()
        finals = finals[finals["document_type"].isin(DOC_TYPES)]
        finals = finals[finals["configuration"].isin(["Baseline", "SAFE_OBSERVED", "FASTEST_EXPERIMENTAL"])]
        self.make_plot("01_final_throughput", "Final full-pipeline throughput", "Docs per second for baseline, safe observed, and fastest experimental configurations.", "Final integrated runs are measured combined configurations; no speedup is claimed for missing MRZ-only baselines.", lambda fig, ax: (self.grouped_bars(ax, finals.assign(configuration=finals.configuration.replace({"Baseline": "Baseline"})), "document_type", "docs_per_second", "configuration", DOC_TYPES), ax.set_ylabel("documents / second")))
        seconds = finals.copy(); seconds["configuration"] = seconds["configuration"].replace({"Baseline": "Baseline"})
        self.make_plot("01b_final_seconds_per_document", "Final full-pipeline seconds per document", "The same final runs expressed as seconds per logical document.", "Lower is better; ID cards remain one logical document despite two physical sides.", lambda fig, ax: (self.grouped_bars(ax, seconds.assign(seconds_per_doc=seconds["seconds"] / seconds["document_type"].map({k: len([a for a in self.annotations.values() if a.get("document_type") == k]) for k in DOC_TYPES})), "document_type", "seconds_per_doc", "configuration", DOC_TYPES), ax.set_ylabel("seconds / logical document")))
        baseline = finals[finals.configuration == "Baseline"].groupby("document_type")["median_seconds"].median().to_dict()
        speed_rows = []
        for _, row in finals[finals.configuration != "Baseline"].groupby(["document_type", "configuration"], as_index=False).median(numeric_only=True).iterrows():
            base = baseline.get(row.document_type)
            if base:
                speed_rows.append({"document_type": row.document_type, "configuration": row.configuration, "speedup": base / row.median_seconds})
        speed = pd.DataFrame(speed_rows)
        self.make_plot("02_speedup_vs_baseline", "E2E speedup versus baseline", "Measured final-config speedup from the same full-pipeline baseline scope.", "Only final full-pipeline rows with a legitimate 01.baseline counterpart are included.", lambda fig, ax: (ax.axvline(1, color="black", lw=1), ax.barh([f"{DISPLAY_TYPES[r.document_type]} — {r.configuration}" for _, r in speed.iterrows()], speed.speedup, color=[COLORS.get(r.configuration, "#777") for _, r in speed.iterrows()]), ax.set_xlabel("speedup (baseline median / candidate median)")))

    def plot_accuracy(self, accuracy: pd.DataFrame) -> None:
        visible = accuracy[accuracy.configuration.isin(["Baseline", "SAFE_OBSERVED", "FASTEST_EXPERIMENTAL", "V0_MEDIUM_BASELINE", "V1_FAST_ONLY"])].copy()
        def visible_plot(fig, ax):
            exact = visible.pivot_table(index="document_type", columns="configuration", values="exact_rate", aggfunc="first").reindex(DOC_TYPES)
            chars = visible.pivot_table(index="document_type", columns="configuration", values="character_accuracy", aggfunc="first").reindex(DOC_TYPES)
            exact.plot.bar(ax=ax, width=.8, color=[COLORS.get(str(c), "#888888") for c in exact.columns], title="Field exact-match rate")
            ax.set_ylabel("rate"); ax.set_ylim(0, 1.05); ax.set_xticks(range(len(exact.index)), [DISPLAY_TYPES[k] for k in exact.index], rotation=0); ax.legend(fontsize=7)
            second = ax.twinx()
            chars.plot(ax=second, marker="o", linewidth=1.5, color=[COLORS.get(str(c), "#888888") for c in chars.columns], linestyle="--", legend=False)
            second.set_ylim(0, 1.05); second.set_ylabel("character accuracy (dashed)")
        self.make_plot("03_visible_accuracy_overview", "Visible OCR exact match and character accuracy", "Field exact-match rate (bars) and character accuracy (dashed lines) are shown separately.", "Rows come from different experiment paths; the source column in accuracy_summary.csv identifies them. The plot is corpus-level, not a claim about production accuracy.", visible_plot)
        mrz = accuracy[(accuracy.MRZ_documents.fillna(0) > 0) & (accuracy.MRZ_lines.fillna(0) > 0)].copy()
        rows = []
        for _, r in mrz.iterrows():
            rows.extend([{ "document_type": r.document_type, "configuration": r.configuration, "metric": "full MRZ exact", "rate": (r.MRZ_full_exact or 0) / (r.MRZ_documents or 1)}, {"document_type": r.document_type, "configuration": r.configuration, "metric": "line exact", "rate": (r.MRZ_line_exact or 0) / (r.MRZ_lines or 1)}, {"document_type": r.document_type, "configuration": r.configuration, "metric": "character accuracy", "rate": r.character_accuracy}])
        frame = pd.DataFrame(rows)
        self.make_plot("04_mrz_accuracy_overview", "MRZ OCR accuracy: full, line, and character metrics", "Full-MRZ exact, line exact, and character accuracy remain separate.", "Parser and check-digit validation are not treated as OCR correctness; unavailable values remain unavailable.", lambda fig, ax: (frame.pivot_table(index=["document_type", "configuration"], columns="metric", values="rate").plot.bar(ax=ax), ax.set_ylabel("rate"), ax.set_ylim(0, 1.05), ax.tick_params(axis="x", rotation=70)))

    def plot_batch(self, timings: pd.DataFrame) -> None:
        frame = timings[(timings.experiment == "02.recognition-batch") & timings.configuration.str.startswith("batch=")].copy()
        frame["batch"] = frame.configuration.str.replace("batch=", "", regex=False).astype(int)
        for kind in DOC_TYPES:
            part = frame[frame.document_type == kind]
            self.make_plot(f"05_batch_lines_{kind}", f"Experiment 1 — {DISPLAY_TYPES[kind]} batch size vs recognition throughput", "Measured exploratory and repeat points for recognition batch size.", "Only one exploratory point exists for most batch sizes; selected points have repeat rows.", lambda fig, ax, part=part: (ax.plot(part.batch, part.lines_per_second, "o-", label="lines/s") if part.lines_per_second.notna().any() else ax.plot(part.batch, part.docs_per_second, "o-", label="docs/s"), ax.axvline(32, color="grey", ls="--", label="original production batch 32"), ax.set_xlabel("recognition batch size"), ax.set_ylabel("recognition lines/s"), ax.legend()))
            self.make_plot(f"06_batch_e2e_{kind}", f"Experiment 1 — {DISPLAY_TYPES[kind]} batch size vs E2E seconds", "Measured E2E seconds for each recognition batch size.", "This is the visible benchmark path, not a full MRZ-inclusive baseline for every document type.", lambda fig, ax, part=part: (ax.plot(part.batch, part.seconds, "o-"), ax.axvline(32, color="grey", ls="--"), ax.set_xlabel("recognition batch size"), ax.set_ylabel("seconds")))

    def plot_split(self, timings: pd.DataFrame) -> None:
        frame = timings[timings.experiment == "03.split-visible-mrz"].copy()
        frame["configuration"] = frame.configuration.astype(str)
        self.make_plot("07_split_total_seconds", "Experiment 2 — combined versus split OCR", "Total measured seconds for current combined and split modes.", "Split modes are harness comparisons; output correctness is aggregate-only and full paired text is unavailable.", lambda fig, ax: (self.grouped_bars(ax, frame, "document_type", "seconds", "configuration", ["passport", "id_card"]), ax.set_ylabel("seconds")))
        stages = []
        for _, row in frame.iterrows():
            for stage in STAGES:
                value = row.stages.get(stage, 0) if isinstance(row.stages, dict) else 0
                if stage == "other":
                    value = row.stages.get("other", 0)
                stages.append({"document_type": row.document_type, "configuration": row.configuration, "repeat": row.repeat, "stage": stage, "seconds": value})
        stage_frame = pd.DataFrame(stages).groupby(["document_type", "configuration", "stage"], as_index=False).seconds.median()
        self.make_plot("07b_split_stage_time", "Experiment 2 — split stage-time comparison", "Median stage timing for combined and split modes.", "Stage labels follow the benchmark artifact schema; tiny bookkeeping stages may be absorbed into other.", lambda fig, ax: self._stacked(ax, stage_frame, "document_type", "configuration", "seconds"))
        saved = []
        for kind in ("passport", "id_card"):
            part = frame[frame.document_type == kind].groupby("configuration").seconds.median()
            current = part.get("CURRENT_COMBINED")
            for config in ("SPLIT_SAME_BATCH", "SPLIT_TUNED_BATCH"):
                if current is not None and config in part:
                    saved.append({"document_type": kind, "configuration": config, "seconds_saved": current - part[config], "percent_saved": 100 * (current - part[config]) / current})
        pd.DataFrame(saved).to_csv(self.tables / "split_savings.csv", index=False)

    def _stacked(self, ax, frame: pd.DataFrame, x: str, hue: str, y: str) -> None:
        if frame.empty:
            ax.text(.5, .5, "No stage data", ha="center", va="center"); return
        pivot = frame.pivot_table(index=[x, hue], columns="stage", values=y, aggfunc="median").fillna(0)
        labels = [f"{DISPLAY_TYPES.get(i[0], i[0])}\n{i[1]}" for i in pivot.index]
        bottom = np.zeros(len(pivot))
        for stage in STAGES:
            if stage in pivot:
                values = pivot[stage].to_numpy()
                ax.bar(labels, values, bottom=bottom, label=stage)
                bottom += values
        ax.set_ylabel("seconds"); ax.tick_params(axis="x", rotation=60); ax.legend(fontsize=8)

    def plot_models(self, timings: pd.DataFrame) -> None:
        frame = timings[timings.experiment == "04.recognizer-models"].copy()
        if frame.empty:
            return
        med = frame.groupby("configuration", as_index=False).agg(lines_per_second=("lines_per_second", "median"), milliseconds_per_line=("milliseconds_per_line", "median"), exact_text_match_rate=("exact_text_match_rate", "median"))
        self.make_plot("08_model_lines_per_second", "Experiment 3 — recognizer model throughput", "Fixed-corpus model throughput across successful model runs.", "This corpus is mixed and fixed; per-document-type model throughput was not emitted.", lambda fig, ax: (med.sort_values("lines_per_second").plot.barh(x="configuration", y="lines_per_second", ax=ax, legend=False), ax.set_xlabel("lines/s")))
        self.make_plot("08b_model_ms_per_line", "Experiment 3 — recognizer model milliseconds per line", "Fixed-corpus model latency per line.", "Lower is faster; the medium fixed-corpus wrapper emitted no result.", lambda fig, ax: (med.sort_values("milliseconds_per_line").plot.barh(x="configuration", y="milliseconds_per_line", ax=ax, legend=False), ax.set_xlabel("ms / line")))
        self.make_plot("08c_model_exact_match", "Experiment 3 — fixed-corpus recognizer exact-match correctness", "Exact text match rate reported by the fixed-crop harness.", "This is the harness corpus metric, not a reconstructed visible/MRZ annotation score.", lambda fig, ax: (med.sort_values("exact_text_match_rate").plot.barh(x="configuration", y="exact_text_match_rate", ax=ax, legend=False), ax.set_xlabel("exact text match rate"), ax.set_xlim(0, 1.05)))
        self.make_plot("08d_model_speed_accuracy", "Experiment 3 — fixed-corpus speed versus exact match", "Throughput versus exact text match for successful fixed-corpus model runs.", "The missing medium point is intentionally not estimated or plotted.", lambda fig, ax: (ax.scatter(med.lines_per_second, med.exact_text_match_rate, s=70), [ax.annotate(r.configuration, (r.lines_per_second, r.exact_text_match_rate), fontsize=8) for _, r in med.iterrows()], ax.set_xlabel("lines/s"), ax.set_ylabel("exact text match rate"), ax.set_ylim(0, 1.05)))

    def plot_threads(self) -> None:
        rows = self.all_jsonl.get("05.cpu-runtime/raw.jsonl", [])
        frame = pd.DataFrame([r for r in rows if r.get("backend") == "paddle" and r.get("status") == "ok"])
        if frame.empty:
            return
        self.make_plot("09_cpu_threads_seconds", "Experiment 4 — CPU thread count versus inference time", "Normal Paddle CPU fixed-line inference timing by thread count.", "Only medians were emitted; individual repeat samples are unavailable for box/strip plots.", lambda fig, ax: (ax.plot(frame.threads, frame.median_seconds, "o-"), ax.axvline(4, color="grey", ls="--", label="baseline threads = 4"), ax.axvline(frame.loc[frame.median_seconds.idxmin(), "threads"], color="#2e8b57", ls=":", label="best measured"), ax.set_xlabel("CPU threads"), ax.set_ylabel("median inference seconds"), ax.legend()))
        self.make_plot("09b_cpu_threads_lines", "Experiment 4 — CPU thread count versus lines/s", "Normal Paddle CPU fixed-line inference throughput by thread count.", "The 16-thread regression is measured on the microbenchmark only; it is not a complete-pipeline result.", lambda fig, ax: (ax.plot(frame.threads, frame.lines_per_second, "o-"), ax.axvline(4, color="grey", ls="--"), ax.set_xlabel("CPU threads"), ax.set_ylabel("lines/s")))
        availability = pd.DataFrame([{"runtime": "Paddle CPU", "status": "measured", "detail": "fixed resized line"}, {"runtime": "CPU HPI", "status": "unavailable", "detail": "ultra-infer dependency missing"}, {"runtime": "GPU/TensorRT", "status": "intentionally not tested", "detail": "CPU-only laptop constraint"}])
        availability.to_csv(self.tables / "runtime_availability.csv", index=False)

    def plot_mrz_rows(self) -> None:
        rows = self.all_jsonl.get("06.mrz-rows/raw.jsonl", [])
        frame = pd.DataFrame([r for r in rows if r.get("variant") and r.get("variant") != "DETECTOR_BASED_MEDIUM"])
        if frame.empty:
            return
        frame["lines_per_second"] = frame["lines_per_second"].fillna(frame["line_count"] / frame["seconds"])
        for kind in ("passport", "id_card"):
            part = frame[frame.kind == kind]
            self.make_plot(f"10_mrz_rows_{kind}", f"Experiment 5 — {DISPLAY_TYPES[kind]} MRZ row splitting", "Recognition time and throughput for fixed-row medium/tiny variants.", "Full reconstructed MRZ accuracy is unavailable for this row-level harness.", lambda fig, ax, part=part: (part.groupby("variant").seconds.median().sort_values().plot.barh(ax=ax), ax.set_xlabel("recognition seconds")))
            pd.DataFrame([{"document_type": kind, "configuration": v, "median_seconds": median(g.seconds), "median_lines_per_second": median(g.lines_per_second), "accuracy": "unavailable"} for v, g in part.groupby("variant")]).to_csv(self.tables / f"mrz_rows_{kind}.csv", index=False)

    def plot_fallback(self) -> None:
        rows = [r for r in self.all_jsonl.get("07.fast-fallback/raw.jsonl", []) if "document_id" in r]
        frame = pd.DataFrame(rows)
        if frame.empty:
            return
        frame["state"] = np.select([frame.false_accept.astype(bool), frame.fallback.astype(bool), frame.valid.astype(bool)], ["valid + wrong (false accept)", "invalid → fallback", "valid + correct"], default="unclassified")
        counts = frame.groupby(["kind", "state"]).size().unstack(fill_value=0)
        self.make_plot("11_mrz_fallback_path", "Experiment 6 — MRZ fast-path acceptance and fallback", "Per-document fast-path states, including false accepts.", "Validation-passing but label-wrong outputs are false accepts; parser/check-digit success alone is not OCR correctness.", lambda fig, ax: (counts.plot.bar(stacked=True, ax=ax, color={"valid + correct": "#2e8b57", "valid + wrong (false accept)": "#d62728", "invalid → fallback": "#7570b3"}), ax.set_ylabel("documents"), ax.set_xticklabels([DISPLAY_TYPES.get(str(x), str(x)) for x in counts.index], rotation=0), ax.legend(fontsize=8)))
        self.make_plot("11b_mrz_fallback_timing", "Experiment 6 — MRZ fast path versus fallback time", "Fast, fallback, and total seconds by anonymized document ID.", "Document IDs are safe anonymized labels; output text is intentionally not included.", lambda fig, ax: (frame.sort_values(["kind", "document_id"]).plot.bar(x="document_id", y=["fast_seconds", "fallback_seconds", "total_seconds"], ax=ax), ax.set_ylabel("seconds"), ax.tick_params(axis="x", rotation=70)))
        frame[["kind", "document_id", "valid", "false_accept", "fallback", "fast_seconds", "fallback_seconds", "total_seconds"]].to_csv(self.tables / "mrz_fallback_documents.csv", index=False)

    def plot_transitions(self, output_rows: dict[tuple[str, str], dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame]:
        aggregates, details = [], []
        for kind in DOC_TYPES:
            left = output_rows.get(("SAFE_OBSERVED", kind), {}).get("outputs")
            right = output_rows.get(("FASTEST_EXPERIMENTAL", kind), {}).get("outputs")
            if left and right:
                aggregate, detail = self.paired_transitions(kind, "SAFE_OBSERVED (medium)", "FASTEST_EXPERIMENTAL (tiny)", left, right)
                aggregate["modality"] = "visible"
                aggregates.append(aggregate); details.append(detail)
                aggregates.append(self.paired_mrz_transitions(kind, left, right))
        agg = pd.concat(aggregates, ignore_index=True) if aggregates else pd.DataFrame()
        detail = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
        if not agg.empty:
            agg.to_csv(self.tables / "paired_regressions.csv", index=False)
        if not detail.empty:
            detail.to_csv(self.data / "paired_transition_details.csv", index=False)
        if not detail.empty:
            pivot = detail.groupby(["document_type", "transition"]).size().unstack(fill_value=0).reindex(columns=["BOTH_CORRECT", "REGRESSION", "IMPROVEMENT", "BOTH_WRONG"], fill_value=0)
            self.make_plot("12_paired_transitions", "Visible OCR paired transitions: medium to Tiny", "The four sample-level transition categories for the same final-corpus outputs.", "This is a paired SAFE_OBSERVED-medium versus FASTEST_EXPERIMENTAL-Tiny comparison; the original baseline stores digests, not per-field text.", lambda fig, ax: (pivot.plot.bar(stacked=True, ax=ax, color=["#2e8b57", "#d62728", "#1f77b4", "#7f7f7f"]), ax.set_ylabel("scorable fields"), ax.set_xticklabels([DISPLAY_TYPES.get(str(x), str(x)) for x in pivot.index], rotation=0)))
            for kind in DOC_TYPES:
                part = detail[detail.document_type == kind]
                self.make_plot(f"13_transition_heatmap_{kind}", f"Error transition heatmap — {DISPLAY_TYPES[kind]}", "A 2×2 baseline-correct/candidate-correct transition heatmap.", "Baseline here means the paired stored SAFE_OBSERVED medium output, not the digest-only original full run.", lambda fig, ax, part=part: self._transition_heatmap(ax, part))
        return agg, detail

    def _heatmap(self, ax, frame: pd.DataFrame, rows: list[str], cols: list[str]) -> None:
        if frame.empty:
            ax.text(.5, .5, "No paired outputs", ha="center", va="center"); return
        values = frame.to_numpy(dtype=float)
        im = ax.imshow(values, cmap="Blues")
        for i in range(values.shape[0]):
            for j in range(values.shape[1]): ax.text(j, i, f"{values[i, j]:.0f}", ha="center", va="center")
        ax.set_xticks(range(len(frame.columns)), [str(x) for x in frame.columns]); ax.set_yticks(range(len(frame.index)), [str(x) for x in frame.index]); ax.set_xlabel("candidate state"); ax.set_ylabel("baseline state"); plt.colorbar(im, ax=ax, fraction=.046)

    def _transition_heatmap(self, ax, detail: pd.DataFrame) -> None:
        counts = detail.transition.value_counts()
        matrix = np.array([[counts.get("BOTH_CORRECT", 0), counts.get("REGRESSION", 0)], [counts.get("IMPROVEMENT", 0), counts.get("BOTH_WRONG", 0)]], dtype=float)
        im = ax.imshow(matrix, cmap="Blues")
        for i in range(2):
            for j in range(2): ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center")
        ax.set_xticks([0, 1], ["Candidate correct", "Candidate wrong"])
        ax.set_yticks([0, 1], ["Baseline correct", "Baseline wrong"])
        ax.set_xlabel("candidate state"); ax.set_ylabel("baseline state"); plt.colorbar(im, ax=ax, fraction=.046)

    def plot_per_field_and_documents(self, output_rows: dict[tuple[str, str], dict[str, Any]]) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Counter]]:
        field_frames, doc_frames, confusions = [], [], {}
        for (variant, kind), row in output_rows.items():
            _, fields, docs, confusion = self.score_outputs(variant, kind, row["outputs"], "visible")
            field_frames.append(fields); doc_frames.append(docs); confusions[f"visible:{kind}:{variant}"] = confusion
            _, _, _, mrz_confusion = self.score_outputs(variant, kind, row["outputs"], "mrz")
            confusions[f"mrz:{kind}:{variant}"] = mrz_confusion
        fields = pd.concat(field_frames, ignore_index=True) if field_frames else pd.DataFrame()
        docs = pd.concat(doc_frames, ignore_index=True) if doc_frames else pd.DataFrame()
        if not fields.empty:
            fields[["exact_rate_low", "exact_rate_high"]] = fields.apply(lambda r: pd.Series(wilson(int(r.exact_correct), int(r.n))), axis=1)
            fields.to_csv(self.tables / "per_field_accuracy.csv", index=False)
            for kind in DOC_TYPES:
                part = fields[fields.document_type == kind]
                self.make_plot(f"14_per_field_{kind}", f"Per-field visible exact rate — {DISPLAY_TYPES[kind]}", "Exact-match rate by field for the scored final configurations.", "Only scorable annotation states are included; each bar carries its sample count in the companion CSV.", lambda fig, ax, part=part: self._field_plot(ax, part))
        if not docs.empty:
            detail = self._transition_detail_for_documents(output_rows)
            if not detail.empty:
                doc_delta = detail.groupby(["document_type", "document_id", "transition"]).size().unstack(fill_value=0)
                for col in ("REGRESSION", "IMPROVEMENT"):
                    if col not in doc_delta: doc_delta[col] = 0
                docs = docs.merge(doc_delta[["REGRESSION", "IMPROVEMENT"]].rename(columns={"REGRESSION": "regressions_vs_baseline", "IMPROVEMENT": "improvements_vs_baseline"}).reset_index(), on=["document_type", "document_id"], how="left", suffixes=("", "_paired"))
                docs["regressions_vs_baseline"] = np.where(docs["configuration"].eq("FASTEST_EXPERIMENTAL"), docs["regressions_vs_baseline_paired"].fillna(0), 0).astype(int)
                docs["improvements_vs_baseline"] = np.where(docs["configuration"].eq("FASTEST_EXPERIMENTAL"), docs["improvements_vs_baseline_paired"].fillna(0), 0).astype(int)
                docs = docs.drop(columns=[c for c in ("regressions_vs_baseline_paired", "improvements_vs_baseline_paired") if c in docs])
            docs.to_csv(self.tables / "per_document.csv", index=False)
            for kind in DOC_TYPES:
                part = docs[docs.document_type == kind]
                self.make_plot(f"15_document_difficulty_{kind}", f"Document-level visible accuracy — {DISPLAY_TYPES[kind]}", "Per-document field accuracy for medium and Tiny final outputs.", "These are empirical OCR difficulty proxies, not human image-quality labels.", lambda fig, ax, part=part: (part.pivot(index="document_id", columns="configuration", values="accuracy").plot.bar(ax=ax), ax.set_ylabel("field exact-match rate"), ax.set_ylim(0, 1.05), ax.tick_params(axis="x", rotation=0)))
        return fields, docs, confusions

    def _transition_detail_for_documents(self, output_rows: dict[tuple[str, str], dict[str, Any]]) -> pd.DataFrame:
        frames = []
        for kind in DOC_TYPES:
            left = output_rows.get(("SAFE_OBSERVED", kind), {}).get("outputs")
            right = output_rows.get(("FASTEST_EXPERIMENTAL", kind), {}).get("outputs")
            if left and right:
                _, detail = self.paired_transitions(kind, "SAFE_OBSERVED (medium)", "FASTEST_EXPERIMENTAL (tiny)", left, right)
                frames.append(detail)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    def _field_plot(self, ax, part: pd.DataFrame) -> None:
        if part.empty: ax.text(.5, .5, "No scorable fields", ha="center", va="center"); return
        pivot = part.pivot_table(index="field", columns="configuration", values="exact_rate", aggfunc="mean").sort_values(list(part.configuration.unique())[0])
        pivot.plot.barh(ax=ax); ax.set_xlabel("exact-match rate"); ax.set_xlim(0, 1.05); ax.legend(fontsize=8)

    def plot_difficulty(self, docs: pd.DataFrame) -> None:
        if docs.empty: return
        baseline = docs[docs.configuration == "SAFE_OBSERVED"].set_index(["document_type", "document_id"])["accuracy"]
        rows = []
        for key, value in baseline.items():
            bucket = "baseline-easy" if value >= .9 else "baseline-hard" if value < .5 else "baseline-mixed"
            rows.append({"document_type": key[0], "document_id": key[1], "baseline_proxy_accuracy": value, "difficulty_bucket": bucket})
        buckets = pd.DataFrame(rows)
        buckets.to_csv(self.tables / "difficulty_buckets.csv", index=False)
        joined = docs.merge(buckets, on=["document_type", "document_id"])
        summary = joined.groupby(["document_type", "difficulty_bucket", "configuration"], as_index=False).agg(accuracy=("accuracy", "mean"), documents=("document_id", "nunique"))
        summary.to_csv(self.tables / "difficulty_bucket_accuracy.csv", index=False)
        self.make_plot("16_difficulty_buckets", "Empirical difficulty buckets relative to medium output", "Candidate accuracy within baseline-easy, mixed, and hard buckets.", "Buckets use observed SAFE_OBSERVED field accuracy thresholds: easy ≥90%, hard <50%, otherwise mixed; raw document values remain in per_document.csv.", lambda fig, ax: (summary.pivot_table(index=["document_type", "difficulty_bucket"], columns="configuration", values="accuracy").plot.bar(ax=ax), ax.set_ylabel("field exact-match rate"), ax.set_ylim(0, 1.05), ax.tick_params(axis="x", rotation=60)))

    def plot_alignment(self) -> None:
        rows = []
        for path, records in self.all_jsonl.items():
            for record in records:
                samples = record.get("diagnostics", {}).get("line_filter", {}).get("samples", {})
                for key, value in samples.items():
                    if not isinstance(value, dict): continue
                    document_id = next((x for x in self.annotations if x in key), None)
                    rows.append({"source": path, "document_id": document_id or key, "detected_line_count": value.get("detected_line_count"), "recognition_candidate_count": value.get("recognition_candidate_count"), "filtered_before_recognition_count": value.get("filtered_before_recognition_count")})
        frame = pd.DataFrame(rows).dropna(subset=["detected_line_count", "recognition_candidate_count"])
        if frame.empty:
            (self.data / "alignment_availability.txt").write_text("No line-filter/alignment diagnostics were stored. Alignment quality cannot currently be quantified from these artifacts.\n", encoding="utf-8")
            return
        frame.to_csv(self.tables / "alignment_diagnostics.csv", index=False)
        self.make_plot("17_alignment_proxies", "Alignment-related diagnostic proxies", "Stored detected-line and recognition-candidate counts by anonymized sample label.", "These are objective artifact metrics only; they are not labels for good/bad alignment and no causal claim is made.", lambda fig, ax: (frame.groupby("document_id")[["detected_line_count", "recognition_candidate_count"]].median().plot.bar(ax=ax), ax.set_ylabel("count"), ax.tick_params(axis="x", rotation=70)))

    def plot_confusions(self, confusions: dict[str, Counter]) -> None:
        for modality in ("visible", "mrz"):
            combined = Counter()
            for key, value in confusions.items():
                if key.startswith(modality + ":"): combined.update(value)
            rows = [{"from": a, "to": b, "count": n} for (a, b), n in combined.most_common() if a not in {"<", "insertion"} and b not in {"omission", ">"}]
            pd.DataFrame(rows).to_csv(self.tables / f"character_confusions_{modality}.csv", index=False)
            top = pd.DataFrame(rows).head(15)
            self.make_plot(f"18_character_confusions_{modality}", f"Top character substitutions — {modality}", "Aggregate substitutions from stored predictions versus annotation text, with no document text printed.", "Counts pool all scored final outputs; a low count reflects limited corpus evidence, not a production probability.", lambda fig, ax, top=top: (top.assign(label=top["from"] + " → " + top["to"]).sort_values("count").plot.barh(x="label", y="count", ax=ax, legend=False) if not top.empty else ax.text(.5, .5, "No substitutions", ha="center", va="center")))
            chars = sorted(set(top.get("from", [])) | set(top.get("to", []))) if not top.empty else []
            matrix = pd.DataFrame(0, index=chars, columns=chars)
            for _, row in top.iterrows(): matrix.loc[row["from"], row["to"]] = row["count"]
            if not matrix.empty:
                self.make_plot(f"18b_character_confusion_heatmap_{modality}", f"Character confusion heatmap — {modality}", "Common aggregate character substitutions in matrix form.", "Only substitution counts are shown; omissions and insertions are listed in the CSV.", lambda fig, ax, matrix=matrix: self._heatmap(ax, matrix, list(matrix.index), list(matrix.columns)))

    def plot_edit_distance(self, output_rows: dict[tuple[str, str], dict[str, Any]]) -> None:
        rows = []; mrz_rows = []
        for (variant, kind), row in output_rows.items():
            for did, output in row["outputs"].items():
                for field, entry in self.truth_fields(did).items():
                    if not self.scorable(entry): continue
                    expected = "" if entry.get("state") == "empty" else str(entry.get("value") or "")
                    actual = str(output.get("fields", {}).get(field) or "")
                    distance = lev(expected, actual)
                    bucket = "exact" if distance == 0 else "1 character wrong" if distance == 1 else "2 characters wrong" if distance == 2 else "3+ characters wrong"
                    if not actual: bucket = "missing"
                    elif len(actual) > len(expected) and distance > 0: bucket = "extra / 3+ wrong" if distance >= 3 else bucket
                    rows.append({"document_type": kind, "configuration": variant, "distance": distance, "normalized_edit_distance": distance / max(len(expected), 1), "bucket": bucket})
                for index, expected in enumerate(self.annotations.get(did, {}).get("mrz", {}).get("lines", [])):
                    actual = str(output.get("mrz", [])[index] if index < len(output.get("mrz", [])) else "")
                    distance = lev(str(expected), actual)
                    mrz_rows.append({"document_type": kind, "configuration": variant, "distance": distance, "normalized_edit_distance": distance / max(len(str(expected)), 1), "bucket": "exact" if distance == 0 else "1 character wrong" if distance == 1 else "2 characters wrong" if distance == 2 else "3+ characters wrong"})
        frame = pd.DataFrame(rows)
        frame.to_csv(self.tables / "edit_distance_samples.csv", index=False)
        if frame.empty: return
        self.make_plot("19_edit_distance_distribution", "Visible OCR normalized edit-distance distribution", "Sample-level near misses distinguish one-character errors from larger failures.", "Distribution is over scorable fields in the 20-document corpus, not a confidence interval for production.", lambda fig, ax: (frame.boxplot(column="normalized_edit_distance", by="configuration", ax=ax), ax.set_title(""), fig.suptitle("Visible OCR normalized edit distance"), ax.set_ylabel("distance / max(truth length, 1)")))
        order = ["exact", "1 character wrong", "2 characters wrong", "3+ characters wrong", "missing", "extra / 3+ wrong"]
        counts = frame.groupby(["document_type", "configuration", "bucket"]).size().unstack(fill_value=0).reindex(columns=order, fill_value=0)
        self.make_plot("19b_exact_vs_near_miss", "Visible OCR exact match versus near misses", "Stacked counts of exact, 1-character, 2-character, and larger errors.", "Missing and extra text remain distinct categories; source-level normalization follows the benchmark string comparison.", lambda fig, ax: (counts.plot.bar(stacked=True, ax=ax), ax.set_ylabel("fields"), ax.tick_params(axis="x", rotation=70), ax.legend(fontsize=7)))
        mrz_frame = pd.DataFrame(mrz_rows)
        mrz_frame.to_csv(self.tables / "mrz_edit_distance_samples.csv", index=False)
        if not mrz_frame.empty:
            self.make_plot("19c_mrz_edit_distance_distribution", "MRZ normalized edit-distance distribution", "Full reconstructed MRZ-line near misses for stored final outputs.", "Parser/check-digit status is intentionally not substituted for line text comparison.", lambda fig, ax: (mrz_frame.boxplot(column="normalized_edit_distance", by="configuration", ax=ax), ax.set_title(""), fig.suptitle("MRZ line normalized edit distance"), ax.set_ylabel("distance / line length")))

    def plot_stage_and_amdahl(self, timings: pd.DataFrame) -> None:
        frame = timings[(timings.experiment.isin(["01.baseline", "08.final-result"])) & timings.document_type.isin(DOC_TYPES) & timings.configuration.isin(["Baseline", "SAFE_OBSERVED", "FASTEST_EXPERIMENTAL"])].copy()
        stage_rows = []
        for (kind, config), group in frame.groupby(["document_type", "configuration"]):
            row = {"document_type": kind, "configuration": config}
            for stage in STAGES: row[stage] = median(group.stages.map(lambda s: s.get(stage, 0) if isinstance(s, dict) else 0)) or 0
            total = sum(row[s] for s in STAGES)
            if total: stage_rows.append({**row, **{s: row[s] / total for s in STAGES}})
        shares = pd.DataFrame(stage_rows)
        for kind in DOC_TYPES:
            part = shares[shares.document_type == kind]
            self.make_plot(f"20_stage_share_{kind}", f"Measured stage share — {DISPLAY_TYPES[kind]}", "Localization, detection, recognition, MRZ recognition, and other shares for baseline and final configurations.", "Shares are based on measured stage timing; ‘other’ includes residual orchestration time.", lambda fig, ax, part=part: self._stacked_share(ax, part))
        amdahl = []
        for _, row in shares[shares.configuration == "Baseline"].iterrows():
            for stage in ("localization", "detection", "recognition"):
                share = row.get(stage, 0)
                amdahl.append({"document_type": row.document_type, "removed_stage": stage, "theoretical_max_speedup": 1 / max(1 - share, 1e-9)})
        amdahl_frame = pd.DataFrame(amdahl)
        amdahl_frame.to_csv(self.tables / "amdahl_upper_bounds.csv", index=False)
        self.make_plot("21_amdahl_upper_bound", "Theoretical Amdahl upper bound — not measured speedup", "Upper-bound speedup if an entire baseline stage disappeared, using measured stage shares.", "This is a theoretical upper bound and does not predict a real optimization result.", lambda fig, ax: (amdahl_frame.pivot(index="document_type", columns="removed_stage", values="theoretical_max_speedup").plot.bar(ax=ax), ax.set_ylabel("theoretical maximum speedup"), ax.axhline(1, color="black", lw=1), ax.tick_params(axis="x", rotation=0)))

    def plot_repeat_distributions(self, timings: pd.DataFrame) -> None:
        frame = timings[(timings.experiment.isin(["01.baseline", "08.final-result"])) & timings.configuration.isin(["Baseline", "SAFE_OBSERVED", "FASTEST_EXPERIMENTAL"])].copy()
        if frame.empty: return
        self.make_plot("24_timing_distributions", "Repeated full-pipeline timing distributions", "All emitted repeat timings for baseline and final full-pipeline configurations.", "These are the samples emitted by the benchmark; the plot does not manufacture variance for single-median microbenchmarks.", lambda fig, ax: (frame.assign(label=frame.document_type.map(DISPLAY_TYPES) + " — " + frame.configuration).boxplot(column="seconds", by="label", ax=ax, rot=70), ax.set_title(""), fig.suptitle("Full-pipeline timing distributions"), ax.set_ylabel("seconds")))

    def plot_opportunities(self, timings: pd.DataFrame, transitions: pd.DataFrame) -> None:
        baseline = timings[(timings.experiment == "01.baseline") & (timings.configuration == "Baseline")].groupby("document_type").seconds.median()
        final = timings[(timings.experiment == "08.final-result") & timings.configuration.isin(["SAFE_OBSERVED", "FASTEST_EXPERIMENTAL"])].groupby(["document_type", "configuration"]).seconds.median()
        rows = []
        for kind in DOC_TYPES:
            for config in ("SAFE_OBSERVED", "FASTEST_EXPERIMENTAL"):
                if kind in baseline.index and (kind, config) in final.index:
                    reg = transitions.loc[transitions.document_type == kind, "regressions"].iloc[0] if config == "FASTEST_EXPERIMENTAL" and not transitions.loc[transitions.document_type == kind].empty else 0
                    rows.append({"optimization_family": f"{DISPLAY_TYPES[kind]} {config}", "document_type": kind, "configuration": config, "measured_speedup": baseline[kind] / final[(kind, config)], "regression_count": reg})
        # Same-experiment comparisons are retained as opportunities, not combined scores.
        batch = timings[timings.experiment == "02.recognition-batch"].copy()
        for kind, part in batch.groupby("document_type"):
            refs = part[part.configuration == "batch=32"].seconds
            if not refs.empty:
                for config in ("batch=8", "batch=12"):
                    candidate = part[part.configuration == config].seconds
                    if not candidate.empty:
                        rows.append({"optimization_family": f"{DISPLAY_TYPES[kind]} {config} vs batch=32", "document_type": kind, "configuration": config, "measured_speedup": refs.median() / candidate.median(), "regression_count": 0})
        split = timings[timings.experiment == "03.split-visible-mrz"]
        for kind, part in split.groupby("document_type"):
            ref = part[part.configuration == "CURRENT_COMBINED"].seconds
            if not ref.empty:
                for config in ("SPLIT_SAME_BATCH", "SPLIT_TUNED_BATCH"):
                    candidate = part[part.configuration == config].seconds
                    if not candidate.empty: rows.append({"optimization_family": f"{DISPLAY_TYPES[kind]} {config}", "document_type": kind, "configuration": config, "measured_speedup": ref.median() / candidate.median(), "regression_count": np.nan})
        models = timings[timings.experiment == "04.recognizer-models"]
        if not models.empty:
            ref = models[models.configuration == "PP-OCRv6_small_rec"].lines_per_second.median()
            tiny = models[models.configuration == "PP-OCRv6_tiny_rec"].lines_per_second.median()
            if ref and tiny: rows.append({"optimization_family": "Tiny vs Small fixed-crop", "document_type": "fixed_corpus", "configuration": "PP-OCRv6_tiny_rec", "measured_speedup": tiny / ref, "regression_count": np.nan})
        thread_rows = pd.DataFrame(self.all_jsonl.get("05.cpu-runtime/raw.jsonl", []))
        if not thread_rows.empty:
            thread_rows = thread_rows[(thread_rows.backend == "paddle") & (thread_rows.status == "ok")]
            if not thread_rows[thread_rows.threads == 4].empty:
                rows.append({"optimization_family": "CPU threads 12 vs 4", "document_type": "fixed_line", "configuration": "12 threads", "measured_speedup": thread_rows[thread_rows.threads == 4].median_seconds.iloc[0] / thread_rows[thread_rows.threads == 12].median_seconds.iloc[0], "regression_count": np.nan})
        fallback = timings[timings.experiment == "07.fast-fallback"]
        for kind, part in fallback.groupby("document_type"):
            ref = part[part.configuration == "V0_MEDIUM_BASELINE"].docs_per_second
            cand = part[part.configuration == "V1_FAST_ONLY"].docs_per_second
            if not ref.empty and not cand.empty:
                reg = transitions.loc[transitions.document_type == kind, "regressions"].iloc[0] if not transitions.loc[transitions.document_type == kind].empty else np.nan
                rows.append({"optimization_family": f"{DISPLAY_TYPES[kind]} fast-only vs V0", "document_type": kind, "configuration": "V1_FAST_ONLY", "measured_speedup": cand.iloc[0] / ref.iloc[0], "regression_count": reg})
        frame = pd.DataFrame(rows); frame.to_csv(self.tables / "opportunity_summary.csv", index=False)
        if frame.empty: return
        self.make_plot("25_opportunity_speedup", "Measured opportunities by optimization outcome", "Speedup by document type for safe observed and fastest experimental final configurations.", "Speed and regression count are deliberately shown in separate aligned views; no combined score is used.", lambda fig, ax: (frame.plot.barh(x="optimization_family", y="measured_speedup", ax=ax, legend=False), ax.axvline(1, color="black", lw=1), ax.set_xlabel("measured speedup")))
        self.make_plot("25b_opportunity_regressions", "Measured opportunities — paired regression count", "Paired visible regression counts for the final optimization configurations.", "Regression counts require stored paired outputs; the original baseline digest-only path is unavailable at field level.", lambda fig, ax: (frame.plot.barh(x="optimization_family", y="regression_count", ax=ax, legend=False, color="#d62728"), ax.set_xlabel("regression fields")))

    def _stacked_share(self, ax, part: pd.DataFrame) -> None:
        if part.empty: ax.text(.5, .5, "No stage data", ha="center", va="center"); return
        values = part.set_index("configuration")[list(STAGES)]
        values.plot.bar(stacked=True, ax=ax, color=["#4c566a", "#7570b3", "#d95f02", "#1b9e77", "#bdbdbd"])
        ax.set_ylabel("share of measured time"); ax.set_ylim(0, 1.05); ax.tick_params(axis="x", rotation=0); ax.legend(fontsize=8)

    def plot_frontiers(self, timings: pd.DataFrame, accuracy: pd.DataFrame) -> None:
        output_rows = self.output_rows()
        for kind in DOC_TYPES:
            rows = []
            for variant in ("SAFE_OBSERVED", "FASTEST_EXPERIMENTAL"):
                row = output_rows.get((variant, kind))
                if not row: continue
                vis, _, _, _ = self.score_outputs(variant, kind, row["outputs"], "visible")
                rows.append({"configuration": variant, "throughput": len(row["outputs"]) / row["seconds"], "exact_rate": vis.get("exact_rate"), "character_accuracy": vis.get("character_accuracy")})
            frame = pd.DataFrame(rows)
            if frame.empty: continue
            self.make_plot(f"22_frontier_visible_{kind}", f"Visible OCR speed versus exact match — {DISPLAY_TYPES[kind]}", "Full-pipeline docs/s versus field exact-match rate for stored final configurations.", "Pareto interpretation is only for this validation corpus; throughput and correctness are not combined into a score.", lambda fig, ax, frame=frame: (ax.scatter(frame.throughput, frame.exact_rate, s=80), [ax.annotate(r.configuration, (r.throughput, r.exact_rate), fontsize=8) for _, r in frame.iterrows()], ax.set_xlabel("docs/s"), ax.set_ylabel("field exact-match rate"), ax.set_ylim(0, 1.05)))
            self.make_plot(f"22b_frontier_visible_char_{kind}", f"Visible OCR speed versus character accuracy — {DISPLAY_TYPES[kind]}", "Full-pipeline docs/s versus character accuracy.", "Character accuracy can hide field-boundary failures; inspect exact match and transitions together.", lambda fig, ax, frame=frame: (ax.scatter(frame.throughput, frame.character_accuracy, s=80), [ax.annotate(r.configuration, (r.throughput, r.character_accuracy), fontsize=8) for _, r in frame.iterrows()], ax.set_xlabel("docs/s"), ax.set_ylabel("character accuracy"), ax.set_ylim(0, 1.05)))
        mrz_rows = []
        fb = pd.DataFrame([r for r in self.all_jsonl.get("07.fast-fallback/raw.jsonl", []) if "document_id" in r])
        for kind in ("passport", "id_card"):
            part = fb[fb.kind == kind]
            if part.empty: continue
            exact = []
            for _, r in part.iterrows():
                annotation = self.annotations.get(r.document_id, {}); expected = annotation.get("mrz", {}).get("lines", []); actual = r.get("output", {}).get("mrz", [])
                exact.append(int(len(expected) == len(actual) and all(a == b for a, b in zip(expected, actual))))
            mrz_rows.append({"document_type": kind, "configuration": "fast path", "throughput": 1 / part.total_seconds.median(), "full_exact_rate": sum(exact) / len(exact), "false_accepts": int(part.false_accept.sum())})
        mrz = pd.DataFrame(mrz_rows)
        if not mrz.empty:
            self.make_plot("23_frontier_mrz", "MRZ speed versus full-MRZ exact match", "Fast-path MRZ throughput versus full reconstructed MRZ exact match.", "Original MRZ-only baseline timing is absent; this is a candidate point, not a baseline speedup claim. False accepts are annotated in the CSV.", lambda fig, ax: (ax.scatter(mrz.throughput, mrz.full_exact_rate, s=80), [ax.annotate(f"{r.configuration} ({r.false_accepts} false accepts)", (r.throughput, r.full_exact_rate), fontsize=8) for _, r in mrz.iterrows()], ax.set_xlabel("MRZ docs/s"), ax.set_ylabel("full-MRZ exact-match rate"), ax.set_ylim(0, 1.05)))
        # Pareto table on final visible points.
        frontier = []
        for kind in DOC_TYPES:
            points = []
            for variant in ("SAFE_OBSERVED", "FASTEST_EXPERIMENTAL"):
                row = output_rows.get((variant, kind))
                if row:
                    metric, _, _, _ = self.score_outputs(variant, kind, row["outputs"], "visible")
                    points.append((variant, len(row["outputs"]) / row["seconds"], metric["exact_rate"]))
            for config, speed, acc in points:
                dominated = any(other != config and s >= speed and a >= acc and (s > speed or a > acc) for other, s, a in points)
                frontier.append({"document_type": kind, "configuration": config, "throughput": speed, "exact_rate": acc, "pareto_dominated": dominated})
        pd.DataFrame(frontier).to_csv(self.tables / "pareto_frontier.csv", index=False)

    def write_report(self, timings: pd.DataFrame, accuracy: pd.DataFrame, transitions: pd.DataFrame, docs: pd.DataFrame) -> None:
        coverage = self.annotations_coverage()
        speed = timings[(timings.experiment == "08.final-result") & timings.document_type.isin(DOC_TYPES) & timings.configuration.isin(["SAFE_OBSERVED", "FASTEST_EXPERIMENTAL"])].groupby(["document_type", "configuration"]).seconds.median().unstack()
        lines = ["# Optimization-six visual analysis", "", "## Executive interpretation", "", "All conclusions below are limited to this small validation corpus: 9 passports, 4 logical ID cards / 8 sides, and 7 driving licences. Absolute rates are therefore evidence for paired investigation, not production guarantees.", ""]
        if not speed.empty:
            for kind in DOC_TYPES:
                if kind in speed.index:
                    vals = speed.loc[kind]
                    lines.append(f"- **{DISPLAY_TYPES[kind]} performance:** " + ", ".join(f"{k} {v:.2f}s ({len([a for a in self.annotations.values() if a.get('document_type') == kind]) / v:.3f} docs/s)" for k, v in vals.items() if pd.notna(v)) + ".")
        if not transitions.empty:
            for _, r in transitions.iterrows():
                if r.get("modality", "visible") == "visible":
                    lines.append(f"- **{DISPLAY_TYPES[r.document_type]} paired medium → Tiny:** {int(r.regressions)} regressions, {int(r.improvements)} improvements, {int(r.both_wrong)} both-wrong, net {int(r.net_change)} fields. Most of the remaining errors are shared only when `BOTH_WRONG` is large; regressions are the new-candidate risk.")
        lines += ["", "## Dataset quality / evaluation coverage", "", markdown_table(coverage), "", "Scorable fields are annotation states `value` plus `empty`; `unreadable` and unannotated states are not converted into OCR failures. The current files contain no unreadable fields, but the loader preserves that category.", "", "## Performance", "", "![Final throughput](plots/01_final_throughput.png)", "", "![Speedup](plots/02_speedup_vs_baseline.png)", "", "![Batch sweep](plots/06_batch_e2e_passport.png)", "", "Batch 8 is the visible-path measured winner in the existing report for passport and driving, while larger batches regress on this CPU. Repeat distributions are only available where raw repeated timings exist; several microbenchmarks emitted medians only.", "", "## Accuracy", "", "![Visible accuracy](plots/03_visible_accuracy_overview.png)", "", "![MRZ accuracy](plots/04_mrz_accuracy_overview.png)", "", "The original full baseline has aggregate correctness but no per-document prediction text, so original-baseline per-field transitions cannot be reconstructed. The script instead provides exact paired transitions for stored SAFE_OBSERVED medium versus FASTEST_EXPERIMENTAL Tiny outputs, and keeps original aggregate baseline rows in `accuracy_summary.csv`.", "", "## Speed-vs-accuracy trade-off", "", "![Passport frontier](plots/22_frontier_visible_passport.png)", "", "![ID frontier](plots/22_frontier_visible_id_card.png)", "", "![Driving frontier](plots/22_frontier_visible_driving_license.png)", "", "Pareto labels describe the current corpus only. No weighted score was created.", "", "## Where candidates regress", "", "![Transitions](plots/12_paired_transitions.png)", "", "The transition heatmaps and `data/paired_transition_details.csv` allow field/document drill-down without printing any recognized PII.", "", "## MRZ-specific findings", "", "![Fallback](plots/11_mrz_fallback_path.png)", "", "Fast/fallback artifacts report the exact acceptance and fallback states, including 4 false accepts. Those false-valid document IDs are in `tables/mrz_fallback_documents.csv`; raw MRZ strings are deliberately not copied into this report. Original MRZ-only baseline timing is unavailable, and Experiment 5 did not reconstruct complete candidate MRZ lines for annotation-scored accuracy.", "", "## Per-field findings", "", "![Passport fields](plots/14_per_field_passport.png)", "", "![Driving fields](plots/14_per_field_driving_license.png)", "", "Field sample counts and Wilson interval-ready counts are in `tables/per_field_accuracy.csv`; tiny n values should not be read as stable percentages.", "", "## Document difficulty", "", "![Difficulty](plots/16_difficulty_buckets.png)", "", "The difficulty buckets are empirical relative-to-medium proxies: easy ≥90% field accuracy, hard <50%, mixed otherwise. They are not image-quality labels. The raw per-document values are in `tables/per_document.csv`.", "", "## Timing / bottlenecks", "", "![Stage shares](plots/20_stage_share_passport.png)", "", "![Amdahl](plots/21_amdahl_upper_bound.png)", "", "Recognition remains the largest measured stage in the stored final runs where stage data exists. The Amdahl chart is an upper bound, not a measured speedup. CPU HPI is unavailable because `ultra-infer` is missing; GPU/TensorRT was intentionally not tested.", "", "## Conclusions", "", "1. Low absolute accuracy is partly explained by difficult shared failures, but Tiny also introduces measurable new regressions; use the transition counts, not totals alone.", "2. Small/Tiny speed should be treated as a frontier question. Tiny is dramatically faster in the fixed-crop harness, but the final-corpus paired outputs show a substantial visible accuracy loss, especially for driving licence.", "3. Splitting and deterministic row splitting have measured timing benefits, but complete paired MRZ reconstruction is missing for the row-level path, so no safety conclusion is supported.", "4. Next investigation: emit per-document outputs from the fixed-crop and split harnesses, then score visible and reconstructed MRZ text against annotations in the same run. This is the missing evidence needed to distinguish alignment failures from recognizer regressions more sharply.", "", "## Evidence and missing data", "", "- `data/artifact_inventory.csv` inventories every JSON, JSONL, and CSV under the six-experiment root.", "- Medium fixed-crop result: unavailable because the wrapper emitted no result file; it was not estimated.", "- Per-repeat sample distributions for Experiment 4: unavailable because only medians were emitted.", "- Per-document original-baseline predictions: unavailable because the baseline stores output digests, not text.", "- Alignment proxies: emitted line-filter counts are visualized, but no stored per-document accuracy join exists for those samples; no causal claim is made.", ""]
        lines += ["", "## Direct answers to the investigation questions", "", "- **Dataset difficulty:** the annotations contain 117 passport, 48 ID-card, and 91 driving-licence scorable fields; no unreadable or unannotated fields are currently present. The medium/Tiny paired tables show 39, 2, and 32 `BOTH_WRONG` fields respectively, alongside new regressions, so both shared difficulty and optimization regressions are present.", "- **Fewest new regressions:** among the paired final outputs, Tiny adds 5 ID-card regressions, 17 passport regressions, and 23 driving-licence regressions. A per-field annotation-scored Small comparison is unavailable; its fixed-crop harness exact score is not a substitute.", "- **Tiny on hard documents:** raw document-level and empirical bucket tables show whether degradation is concentrated in medium-hard documents; they should be read together because the original baseline has no per-document text.", "- **Small compromise:** Small is much slower than Tiny in the fixed-crop harness (about 25.5 vs 100.6 lines/s median) and both are reported only on the mixed fixed corpus. No annotation-scored visible-field accuracy was emitted for Small, so this remains the next controlled experiment rather than a conclusion.", "- **Split and batch behavior:** split modes change aggregate measured correctness and timing, but the split raw files do not preserve complete candidate outputs. Batch-size medians and repeat rows show batch 8 helps the passport/driving visible paths while ID is essentially flat against batch 32.", "- **Threads:** 12 threads improve the fixed-line microbenchmark; no complete-pipeline 12-thread run exists, so this is not evidence of an E2E improvement.", "- **MRZ gating danger:** the fast path accepted 10 of 13 documents, fell back on 3, and produced 4 false-valid outputs overall (passport `p_4`, `p_5`, `p_7`; ID card `id_4`). The validation gate is therefore not safe as an OCR-correctness gate.", "- **Compute bottleneck:** measured stage shares show recognition as the dominant consumer in the stored full-pipeline runs; the Amdahl plot shows the ceiling from removing each stage and is explicitly not measured speedup.", ""]
        (self.output / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    def run(self) -> None:
        self.write_observed_tables()
        timings = self.timing_rows()
        timings.to_csv(self.data / "timing_repeats.csv", index=False)
        performance = timings.groupby(["experiment", "document_type", "configuration"], as_index=False).agg(median_seconds=("seconds", "median"), docs_per_second=("docs_per_second", "median"), lines_per_second=("lines_per_second", "median"))
        performance["speedup"] = np.nan
        full_baseline = performance[(performance.experiment == "01.baseline") & (performance.configuration == "Baseline")].set_index("document_type").median_seconds.to_dict()
        for experiment, group in performance.groupby("experiment"):
            if experiment == "08.final-result":
                refs = full_baseline
                mask = (performance.experiment == experiment) & performance.document_type.isin(refs) & performance.configuration.isin(["SAFE_OBSERVED", "FASTEST_EXPERIMENTAL"])
                performance.loc[mask, "speedup"] = performance.loc[mask].apply(lambda r: refs[r.document_type] / r.median_seconds if pd.notna(r.median_seconds) else np.nan, axis=1)
            elif experiment == "03.split-visible-mrz":
                refs = group[group.configuration == "CURRENT_COMBINED"].set_index("document_type").median_seconds.to_dict()
                mask = (performance.experiment == experiment) & performance.document_type.isin(refs)
                performance.loc[mask, "speedup"] = performance.loc[mask].apply(lambda r: refs[r.document_type] / r.median_seconds if pd.notna(r.median_seconds) else np.nan, axis=1)
            elif experiment == "02.recognition-batch":
                for kind, ref in group[group.configuration == "batch=32"].set_index("document_type").median_seconds.items():
                    mask = (performance.experiment == experiment) & (performance.document_type == kind)
                    performance.loc[mask, "speedup"] = performance.loc[mask].apply(lambda r, ref=ref: ref / r.median_seconds if pd.notna(r.median_seconds) else np.nan, axis=1)
        performance.to_csv(self.tables / "performance_summary.csv", index=False)
        accuracy = self.baseline_correctness()
        output_rows = self.output_rows()
        transitions, _ = self.plot_transitions(output_rows)
        fields, docs, confusions = self.plot_per_field_and_documents(output_rows)
        self.plot_performance(timings)
        self.plot_accuracy(accuracy)
        self.plot_batch(timings)
        self.plot_split(timings)
        self.plot_models(timings)
        self.plot_threads()
        self.plot_mrz_rows()
        self.plot_fallback()
        self.plot_difficulty(docs)
        self.plot_alignment()
        self.plot_confusions(confusions)
        self.plot_edit_distance(output_rows)
        self.plot_stage_and_amdahl(timings)
        self.plot_frontiers(timings, accuracy)
        self.plot_repeat_distributions(timings)
        self.plot_opportunities(timings, transitions)
        # A compact machine-readable summary of unavailable items.
        availability = pd.DataFrame([
            {"item": "04.recognizer-models: medium fixed-crop measurement", "status": "unavailable", "reason": "wrapper emitted no result file"},
            {"item": "04.recognizer-models: per-document-type model accuracy", "status": "unavailable", "reason": "fixed corpus was emitted as one mixed corpus; no per-corpus annotation join"},
            {"item": "06.mrz-rows: annotation-scored full MRZ accuracy", "status": "unavailable", "reason": "row-level output did not reconstruct complete MRZ comparison"},
            {"item": "01.baseline: per-document prediction text", "status": "unavailable", "reason": "raw baseline stores output digests only"},
            {"item": "05.cpu-runtime: repeat distribution", "status": "unavailable", "reason": "raw file stores medians, not individual repeat samples"},
            {"item": "CPU HPI performance", "status": "unavailable", "reason": "ultra-infer dependency missing; no performance claim made"},
        ])
        availability.to_csv(self.tables / "availability.csv", index=False)
        table_index = ["# Derived tables", "", "Each CSV has a matching Markdown rendering for quick inspection.", ""]
        for path in sorted(self.tables.glob("*.csv")):
            try:
                frame = pd.read_csv(path)
            except Exception:
                continue
            md_path = path.with_suffix(".md")
            md_path.write_text(f"# {path.stem}\n\n{markdown_table(frame)}\n", encoding="utf-8")
            table_index.append(f"- [{path.name}]({path.name}) · [{md_path.name}]({md_path.name})")
        (self.tables / "INDEX.md").write_text("\n".join(table_index) + "\n", encoding="utf-8")
        self.write_report(timings, accuracy, transitions, docs)
        index = ["# Plot index", "", "Every figure is saved as both PNG (180 dpi) and SVG. Derived data and caveats are in the report and tables.", ""]
        for item in self.plot_index:
            index += [f"## [{item['filename']}]({item['filename']})", "", f"- What: {item['what']}", f"- How to interpret: {item['interpretation']}", f"- Caveat: {item['caveat']}", ""]
        (self.plots / "INDEX.md").write_text("\n".join(index), encoding="utf-8")
        summary = {"plots_png": len(self.plot_index), "plots_total_with_svg": len(self.plot_index) * 2, "report": str(self.output / "REPORT.md"), "plot_index": str(self.plots / "INDEX.md"), "missing_data": availability.to_dict(orient="records")}
        (self.data / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs/benchmarks/04.six-optimization-comparison/20260814T000000Z"))
    parser.add_argument("--annotations", type=Path, default=Path("dataset/annotations"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = args.output or args.root / "09.visual-analysis"
    Analysis(args.root, args.annotations, output).run()


if __name__ == "__main__":
    main()
