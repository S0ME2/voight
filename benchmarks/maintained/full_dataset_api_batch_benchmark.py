"""Sweep the v1 batch API with the complete annotated dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import time
import zipfile
from io import BytesIO
from pathlib import Path
from typing import Any

import requests


KINDS = ("passport", "id-card", "driving-license")
FOLDERS = {"passport": "passport", "id-card": "id_card", "driving-license": "driving_license"}


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 7, 8, 9, 16, 32, 64])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 1 for size in args.sizes):
        parser.error("--repeats and --sizes must be positive")
    return args


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _records(root: Path, kind: str) -> list[dict[str, Any]]:
    folder = root / "annotations" / FOLDERS[kind]
    records = []
    for annotation_path in sorted(folder.glob("*.json")):
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        images = {
            role: root / relative
            for role, relative in annotation["images"].items()
        }
        for role, path in images.items():
            if not path.is_file() or _sha256(path) != annotation["image_sha256"][role]:
                raise RuntimeError(f"dataset hash mismatch: {annotation_path}")
        records.append({"id": annotation["id"], "images": images, "fields": annotation["fields"]})
    if not records:
        raise RuntimeError(f"no {kind} records found under {folder}")
    return records


def _payload(kind: str, records: list[dict[str, Any]], count: int) -> tuple[dict[str, Any] | list[tuple[str, tuple[str, bytes, str]]], dict[str, Any]]:
    selected = list(itertools.islice(itertools.cycle(records), count))
    manifest = {
        "requested_logical_documents": count,
        "source_logical_documents": len(records),
        "source_ids": [record["id"] for record in records],
        "selected_ids": [record["id"] for record in selected],
        "physical_images": count * (2 if kind == "id-card" else 1),
    }
    if kind == "id-card":
        archive = BytesIO()
        with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zipped:
            for index, record in enumerate(selected):
                directory = f"card-{index:04d}-{record['id']}"
                for role, path in record["images"].items():
                    zipped.writestr(f"{directory}/{role}{path.suffix.lower()}", path.read_bytes())
        return {"archive": ("cards.zip", archive.getvalue(), "application/zip")}, manifest
    files = []
    for index, record in enumerate(selected):
        path = record["images"]["image"]
        content_type = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        files.append(("images", (f"{index:04d}-{record['id']}{path.suffix.lower()}", path.read_bytes(), content_type)))
    return files, manifest


def _post(base_url: str, kind: str, records: list[dict[str, Any]], count: int, timeout: float) -> dict[str, Any]:
    files, manifest = _payload(kind, records, count)
    started = time.perf_counter()
    try:
        response = requests.post(f"{base_url.rstrip('/')}/v1/ocr/{kind}/batch", files=files, timeout=timeout)
        client_seconds = time.perf_counter() - started
        payload = response.json()
    except Exception as error:
        return {"status": "request_failed", "failure_detail": f"{type(error).__name__}: {error}", **manifest}
    row = {
        "status": "ok" if response.ok and payload.get("failed", 0) == 0 else "request_failed",
        "document_type": kind,
        "document_count": count,
        "client_wall_seconds": client_seconds,
        "server_wall_seconds": payload.get("total_seconds"),
        "documents_per_second": count / client_seconds if client_seconds else None,
        "succeeded": payload.get("succeeded"),
        "failures": payload.get("failed"),
        "failure_detail": None if response.ok else response.text[:1000],
        "localization": payload.get("diagnostics", {}).get("localization", {}),
        "text_detection": payload.get("diagnostics", {}).get("text_detection", {}),
        "text_recognition": payload.get("diagnostics", {}).get("text_recognition", {}),
        **manifest,
    }
    return row


def main() -> None:
    args = _args()
    records = {kind: _records(args.dataset_root, kind) for kind in KINDS}
    warmup = [_post(args.base_url, kind, values, len(values), args.timeout) for kind, values in records.items()]
    rows = []
    for size in args.sizes:
        for kind in KINDS:
            rows.extend(_post(args.base_url, kind, records[kind], size, args.timeout) for _ in range(args.repeats))
    report = {
        "method": "complete CPU-benchmark dataset; deterministic cyclic selection for sizes above each source corpus",
        "dataset": {kind: {"logical_documents": len(values), "physical_images": sum(len(record["images"]) for record in values), "ids": [record["id"] for record in values]} for kind, values in records.items()},
        "sizes": args.sizes,
        "repeats": args.repeats,
        "warmup": warmup,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    with args.output.with_suffix(".csv").open("w", newline="", encoding="utf-8") as output:
        fields = ("document_type", "document_count", "client_wall_seconds", "server_wall_seconds", "documents_per_second", "succeeded", "failures", "status", "source_logical_documents", "physical_images")
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row.get(field) for field in fields} for row in rows)
    print(args.output)


if __name__ == "__main__":
    main()
