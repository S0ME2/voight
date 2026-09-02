"""Normalize the fresh run's side-by-side comparison artifacts."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path


def read(path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write(path, rows):
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        out = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        out.writeheader(); out.writerows(rows)


def main(root: Path) -> None:
    rows = read(root / "field_comparison.csv")
    pairs = {}
    for row in rows:
        pairs[(row["document_id"], row["field"], row["document_type"]), row["candidate"]] = row
    comparison = []
    for key in sorted({key for key, _ in pairs}):
        left, right = pairs[key, "LATIN_OLD"], pairs[key, "MATCHING_NEW"]
        latin_ok = left["status"] in {"match", "likely_match"}
        current_ok = right["status"] in {"match", "likely_match"}
        winner = "Both" if latin_ok and current_ok else "Latin" if latin_ok else "Current" if current_ok else "Both fail"
        comparison.append({"document_type": key[2], "document_id": key[0], "field": key[1], "expected": left["expected"], "latin_detected": left["detected"], "latin_status": left["status"], "latin_correct": latin_ok, "current_detected": right["detected"], "current_status": right["status"], "current_correct": current_ok, "winner": winner})
    write(root / "field_comparison.csv", comparison)

    raw = read(root / "raw_runs.csv")
    ocr = {}
    for row in raw:
        if row.get("phase") == "ocr" and row.get("pass") == "1":
            ocr.setdefault((row["candidate"], row["document_id"]), json.loads(row["response"]))
    failures = read(root / "failure_transitions.csv")
    known = {("p_3", "type"), ("p_7", "name"), ("p_9", "passport_number"), ("d_3", "surname"), ("d_6", "expiry_date"), ("d_6", "personal_id"), ("d_6", "serial_number")}
    for row in failures:
        if (row["document_id"], row["field"]) not in known:
            continue
        for candidate, label in (("LATIN_OLD", "latin_ocr_evidence"), ("MATCHING_NEW", "current_ocr_evidence")):
            payload = ocr.get((candidate, row["document_id"]), {})
            lines = payload.get("lines", payload.get("front", []) + payload.get("back", []))
            row[label] = json.dumps(lines, ensure_ascii=False, separators=(",", ":"))
    write(root / "known_failure_comparison.csv", [row for row in failures if (row["document_id"], row["field"]) in known])

    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    summary["field_transition_counts"] = {row["transition"]: sum(item["transition"] == row["transition"] for item in failures) for row in failures}
    (root / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
