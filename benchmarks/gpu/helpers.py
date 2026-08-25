"""Pure helpers for GPU benchmark artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable


def stats(values: Iterable[float]) -> dict[str, float | None]:
    values = sorted(float(value) for value in values)
    if not values:
        return {"median": None, "iqr": None, "min": None, "max": None}
    quartiles = statistics.quantiles(values, n=4, method="inclusive") if len(values) > 1 else [values[0]] * 3
    return {"median": statistics.median(values), "iqr": quartiles[2] - quartiles[0], "min": values[0], "max": values[-1]}


def semantic_signature(item: dict[str, Any]) -> dict[str, Any]:
    result = item.get("result") or {}
    fields = result.get("fields") or {}
    return {
        "success": bool(item.get("success")),
        "fields": {name: (entry or {}).get("value") for name, entry in sorted(fields.items())},
        "mrz": (result.get("mrz") or {}).get("raw_lines", []),
    }


def semantic_digest(payload: dict[str, Any]) -> str:
    value = [semantic_signature(item) for item in payload.get("items", [])]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _confidence_signature(item: dict[str, Any]) -> dict[str, Any]:
    fields = (item.get("result") or {}).get("fields") or {}
    return {name: (entry or {}).get("confidence") for name, entry in sorted(fields.items())}


def compare_semantics(baseline: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    left = [semantic_signature(item) for item in baseline.get("items", [])]
    right = [semantic_signature(item) for item in candidate.get("items", [])]
    semantic = []
    confidence_only = []
    for index in range(max(len(left), len(right))):
        before, after = left[index] if index < len(left) else None, right[index] if index < len(right) else None
        if before != after:
            semantic.append({"index": index, "baseline": before, "candidate": after})
    for index, (before_item, after_item) in enumerate(zip(baseline.get("items", []), candidate.get("items", []))):
        if semantic_signature(before_item) == semantic_signature(after_item) and _confidence_signature(before_item) != _confidence_signature(after_item):
            confidence_only.append(index)
    return {"semantic_changes": semantic, "confidence_only_changes": confidence_only, "semantic_change_count": len(semantic), "confidence_only_change_count": len(confidence_only)}


def _distance(left: str, right: str) -> int:
    row = list(range(len(right) + 1))
    for index, char in enumerate(left, 1):
        current = [index]
        for other_index, other in enumerate(right, 1):
            current.append(min(current[-1] + 1, row[other_index] + 1, row[other_index - 1] + (char != other)))
        row = current
    return row[-1]


def correctness_score(documents: list[dict[str, Any]], payload: dict[str, Any]) -> dict[str, Any]:
    fields = exact = characters = character_total = 0
    mrz_documents = mrz_found = mrz_exact = mrz_lines = mrz_line_exact = 0
    for document, item in zip(documents, payload.get("items", [])):
        annotation_path = Path(str(document["annotation"]))
        truth = json.loads(annotation_path.read_text(encoding="utf-8")) if annotation_path.is_file() else {}
        result = item.get("result") or {}
        actual_fields = {name: (entry or {}).get("value") for name, entry in (result.get("fields") or {}).items()}
        for name, entry in truth.get("fields", {}).items():
            if not isinstance(entry, dict) or entry.get("state") not in {"value", "empty"}:
                continue
            expected = entry.get("value") if entry.get("state") == "value" else None
            expected_text, actual_text = "" if expected is None else str(expected), "" if actual_fields.get(name) is None else str(actual_fields.get(name))
            fields += 1
            character_total += max(len(expected_text), 1)
            characters += max(len(expected_text), 1) - _distance(expected_text, actual_text)
            exact += actual_fields.get(name) == expected or (expected is None and actual_fields.get(name) in (None, ""))
        expected_lines = [line for line in truth.get("mrz", {}).get("lines", []) if isinstance(line, str)]
        if expected_lines:
            actual_lines = (result.get("mrz") or {}).get("raw_lines", [])
            mrz_documents += 1
            mrz_found += bool(actual_lines)
            mrz_exact += actual_lines == expected_lines
            for index, expected in enumerate(expected_lines):
                actual = actual_lines[index] if index < len(actual_lines) else ""
                mrz_lines += 1
                mrz_line_exact += actual == expected
    return {"items_expected": len(documents), "items_returned": len(payload.get("items", [])), "item_count_difference": len(payload.get("items", [])) - len(documents), "field_correctness": exact / fields if fields else None, "field_character_accuracy": characters / character_total if character_total else None, "mrz_found_rate": mrz_found / mrz_documents if mrz_documents else None, "mrz_exact_match_rate": mrz_exact / mrz_documents if mrz_documents else None, "mrz_line_accuracy": mrz_line_exact / mrz_lines if mrz_lines else None, "field_exact": exact, "field_total": fields, "mrz_found": mrz_found, "mrz_documents": mrz_documents, "mrz_full_exact": mrz_exact, "mrz_line_exact": mrz_line_exact, "mrz_lines": mrz_lines}


def csv_write(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, separators=(",", ":")) if isinstance(value, (dict, list)) else value for key, value in row.items()})


def jsonl_write(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def dataset_manifest(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the same stable document order as the CPU benchmark."""
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    documents: list[dict[str, Any]] = []
    for kind in ("passport", "driving_license"):
        for path in sorted((root / kind).glob("*")):
            if path.is_file() and path.suffix.lower() in extensions:
                documents.append({"document_type": kind, "document_id": path.stem, "paths": [("image", str(path))], "annotation": str(root / "annotations" / kind / f"{path.stem}.json")})
    for directory in sorted((root / "id_card").glob("*")):
        if directory.is_dir():
            paths = []
            for side in ("front", "back"):
                matches = sorted(p for p in directory.glob(f"{side}.*") if p.suffix.lower() in extensions)
                if len(matches) != 1:
                    raise ValueError(f"{directory}: expected one {side} image")
                paths.append((side, str(matches[0])))
            documents.append({"document_type": "id_card", "document_id": directory.name, "paths": paths, "annotation": str(root / "annotations" / "id_card" / f"{directory.name}.json")})
    order = {kind: index for index, kind in enumerate(("passport", "id_card", "driving_license"))}
    documents.sort(key=lambda row: (order[row["document_type"]], row["document_id"].lower()))
    entries = []
    for document in documents:
        images = [{"role": role, "path": path, "sha256": sha256_file(Path(path))} for role, path in document["paths"]]
        annotation = Path(document["annotation"])
        entries.append({"document_id": document["document_id"], "document_type": document["document_type"], "physical_image_count": len(images), "images": images, "annotation_sha256": sha256_file(annotation) if annotation.is_file() else None})
    manifest = {"schema_version": 1, "source_root": str(root), "documents": entries, "counts": {kind: sum(row["document_type"] == kind for row in documents) for kind in order}, "physical_images": sum(len(row["paths"]) for row in documents)}
    return documents, manifest
