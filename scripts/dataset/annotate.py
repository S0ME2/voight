#!/usr/bin/env python3
"""Small, local, document-level ground-truth annotation tool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DOCUMENT_TYPES = ("passport", "id_card", "driving_license")
IMAGE_ROLES = {
    "passport": ("image",),
    "id_card": ("front", "back"),
    "driving_license": ("image",),
}
# These are the current DocumentResult field keys: identity profile names and
# the driving-licence v1 API mapping in app/api/v1.py.
FIELDS = {
    "passport": (
        "type", "country_code", "passport_number", "surname", "name",
        "patronymic", "nationality", "date_of_birth", "sex",
        "place_of_birth", "date_of_issue", "date_of_expiry", "authority",
    ),
    "id_card": (
        "surname", "name", "patronymic", "date_of_birth", "date_of_issue",
        "date_of_expiry", "sex", "citizenship", "card_number", "pinfl",
        "place_of_birth", "place_of_issue",
    ),
    "driving_license": (
        "surname", "given_names", "birth_place", "birth_date", "issue_date",
        "expiry_date", "issued_place", "personal_id", "license_number",
        "address", "categories", "serial_number",
    ),
}
MRZ_SHAPES = {"passport": (2, 44), "id_card": (3, 30)}
STATUSES = {"complete", "in_progress", "skipped"}
FIELD_STATES = {"value", "empty", "unreadable"}
MRZ_RE = re.compile(r"^[A-Z0-9<]+$")


@dataclass(frozen=True)
class Document:
    id: str
    document_type: str
    images: dict[str, str]


def ensure_layout(root: Path) -> None:
    for name in DOCUMENT_TYPES:
        (root / name).mkdir(parents=True, exist_ok=True)
        (root / "annotations" / name).mkdir(parents=True, exist_ok=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, delete=False
        ) as handle:
            temporary = handle.name
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def _images(directory: Path) -> list[Path]:
    return sorted(
        (
            item for item in directory.iterdir()
            if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=lambda item: item.name.lower(),
    ) if directory.is_dir() else []


def discover_documents(root: Path, issues: list[str] | None = None) -> list[Document]:
    found: list[Document] = []
    problems: list[str] = []
    for document_type in ("passport", "driving_license"):
        seen: set[str] = set()
        for path in _images(root / document_type):
            if path.stem in seen:
                problems.append(f"duplicate {document_type} id: {path.stem}")
                continue
            seen.add(path.stem)
            found.append(Document(path.stem, document_type, {"image": path.relative_to(root).as_posix()}))

    directory = root / "id_card"
    if directory.is_dir():
        for card in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
            if not card.is_dir():
                continue
            sides: dict[str, str] = {}
            for side in ("front", "back"):
                matches = [path for path in _images(card) if path.stem.lower() == side]
                if len(matches) != 1:
                    detail = "missing" if not matches else "duplicate"
                    problems.append(f"id_card {card.name}: {detail} {side} image")
                else:
                    sides[side] = matches[0].relative_to(root).as_posix()
            if len(sides) == 2:
                found.append(Document(card.name, "id_card", sides))

    counts = Counter(document.id for document in found)
    problems.extend(f"duplicate logical document id: {identifier}" for identifier, count in counts.items() if count > 1)
    if issues is not None:
        issues.extend(problems)
    elif problems:
        raise ValueError("; ".join(problems))
    return sorted(found, key=lambda item: (DOCUMENT_TYPES.index(item.document_type), item.id.lower()))


def annotation_path(root: Path, document: Document) -> Path:
    return root / "annotations" / document.document_type / f"{document.id}.json"


def load_json(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"annotation must be a JSON object: {path}")
    return data


def new_annotation(root: Path, document: Document) -> dict[str, Any]:
    annotation: dict[str, Any] = {
        "id": document.id,
        "document_type": document.document_type,
        "images": document.images,
        "image_sha256": {
            role: sha256(root / relative) for role, relative in document.images.items()
        },
        "fields": {},
    }
    if document.document_type in MRZ_SHAPES:
        annotation["mrz"] = {"lines": []}
    annotation["status"] = "in_progress"
    return annotation


def annotation_files(root: Path) -> list[Path]:
    base = root / "annotations"
    return sorted(base.glob("**/*.json")) if base.is_dir() else []


def saved_annotations(root: Path) -> tuple[list[dict[str, Any]], list[str]]:
    annotations, errors = [], []
    for path in annotation_files(root):
        try:
            annotations.append(load_json(path))
        except (OSError, json.JSONDecodeError, ValueError) as error:
            errors.append(f"{path.relative_to(root)}: malformed JSON: {error}")
    return annotations, errors


def mrz_warnings(document_type: str, lines: list[Any]) -> list[str]:
    count, width = MRZ_SHAPES[document_type]
    warnings = []
    if len(lines) != count:
        warnings.append(f"expected {count} MRZ lines, got {len(lines)}")
    for index, line in enumerate(lines, 1):
        if not isinstance(line, str):
            warnings.append(f"MRZ line {index} is marked unreadable")
        else:
            if len(line) != width:
                warnings.append(f"MRZ line {index}: expected {width} characters, got {len(line)}")
            if line and not MRZ_RE.fullmatch(line):
                warnings.append(f"MRZ line {index}: contains characters outside A-Z, 0-9, <")
    return warnings


def _field_entry(response: str) -> dict[str, Any]:
    if response == ":empty":
        return {"state": "empty", "value": None}
    if response == ":unreadable":
        return {"state": "unreadable", "value": None}
    return {"state": "value", "value": response}


def _steps(document_type: str) -> list[tuple[str, str | int]]:
    steps: list[tuple[str, str | int]] = [("field", name) for name in FIELDS[document_type]]
    if document_type in MRZ_SHAPES:
        steps.extend(("mrz", index) for index in range(MRZ_SHAPES[document_type][0]))
    steps.append(("notes", "notes"))
    return steps


def _first_unfinished(annotation: dict[str, Any], document_type: str) -> int:
    for index, (kind, key) in enumerate(_steps(document_type)):
        if kind == "field" and key not in annotation.get("fields", {}):
            return index
        if kind == "mrz" and int(key) >= len(annotation.get("mrz", {}).get("lines", [])):
            return index
        if kind == "notes" and "notes" not in annotation:
            return index
    return len(_steps(document_type))


def _current(annotation: dict[str, Any], kind: str, key: str | int) -> Any:
    if kind == "field":
        return annotation.get("fields", {}).get(key)
    if kind == "mrz":
        lines = annotation.get("mrz", {}).get("lines", [])
        return lines[int(key)] if int(key) < len(lines) else None
    return annotation.get("notes")


def _help(output: Callable[[str], None]) -> None:
    output(":q quit and save | :skip mark skipped | :back previous field")
    output(":empty visibly empty/not present | :unreadable cannot read | :help commands")


def show_document(root: Path, document: Document) -> Callable[[], None]:
    """Open a non-blocking OpenCV window; imported lazily for headless tests."""
    import cv2

    panels = []
    for role, relative in document.images.items():
        panel = cv2.imread(str(root / relative))
        if panel is None:
            raise ValueError(f"cannot display image: {relative}")
        scale = min(700 / panel.shape[0], 800 / panel.shape[1], 1.0)
        panel = cv2.resize(panel, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cv2.putText(panel, role.upper(), (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        panels.append(panel)
    height = max(panel.shape[0] for panel in panels)
    panels = [
        cv2.copyMakeBorder(panel, 0, height - panel.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        for panel in panels
    ]
    display = cv2.hconcat(panels)
    title = f"Voight annotation - {document.document_type}: {document.id}"
    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.imshow(title, display)
    cv2.waitKey(1)
    return lambda: cv2.destroyWindow(title)


def annotate_one(
    root: Path,
    document: Document,
    *,
    edit: bool = False,
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
) -> str:
    path = annotation_path(root, document)
    existed = path.exists()
    annotation = load_json(path) if existed else new_annotation(root, document)
    if not existed:
        atomic_json(path, annotation)
    steps = _steps(document.document_type)
    index = 0 if edit else _first_unfinished(annotation, document.document_type)
    if index == len(steps):
        annotation["status"] = "complete"
        atomic_json(path, annotation)
        return "complete"

    while index < len(steps):
        kind, key = steps[index]
        current = _current(annotation, kind, key)
        label = str(key) if kind != "mrz" else f"MRZ line {int(key) + 1}"
        if current is not None:
            output(f"{label} [current: {json.dumps(current, ensure_ascii=False)}]")
        response = input_fn(f"{label}:\n> ")
        if response == ":help":
            _help(output)
            continue
        if response == ":back":
            index = max(0, index - 1)
            continue
        if response == ":q":
            if annotation.get("status") != "complete":
                annotation["status"] = "in_progress"
            atomic_json(path, annotation)
            return "quit"
        if response == ":skip":
            annotation["status"] = "skipped"
            atomic_json(path, annotation)
            return "skipped"

        if kind == "field":
            annotation.setdefault("fields", {})[str(key)] = _field_entry(response)
        elif kind == "mrz":
            value: str | None = None if response == ":unreadable" else "" if response == ":empty" else response
            proposed = list(annotation.setdefault("mrz", {}).setdefault("lines", []))
            while len(proposed) <= int(key):
                proposed.append(None)
            proposed[int(key)] = value
            warnings = mrz_warnings(document.document_type, proposed[: int(key) + 1])
            line_warnings = [warning for warning in warnings if f"line {int(key) + 1}" in warning]
            if line_warnings:
                output("WARNING: " + "; ".join(line_warnings))
                decision = input_fn("Type accept or edit: ").strip().lower()
                if decision != "accept":
                    continue
            annotation["mrz"]["lines"] = proposed
        else:
            if response == ":unreadable":
                output(":unreadable applies to visible fields and MRZ lines")
                continue
            annotation["notes"] = "" if response == ":empty" else response

        annotation["status"] = "in_progress"
        atomic_json(path, annotation)
        index += 1

    annotation["status"] = "complete"
    atomic_json(path, annotation)
    return "complete"


def _status_for(root: Path, document: Document) -> str:
    path = annotation_path(root, document)
    if not path.exists():
        return "unannotated"
    try:
        return str(load_json(path).get("status", "in_progress"))
    except (OSError, json.JSONDecodeError, ValueError):
        return "in_progress"


def startup_summary(root: Path, documents: list[Document]) -> str:
    lines = []
    for document_type in DOCUMENT_TYPES:
        selected = [document for document in documents if document.document_type == document_type]
        counts = Counter(_status_for(root, document) for document in selected)
        lines.extend([
            document_type.replace("_", " ").title() + ":",
            f"  total:       {len(selected)}",
            f"  complete:    {counts['complete']}",
            f"  in progress: {counts['in_progress']}",
            f"  remaining:   {counts['unannotated']}",
            f"  skipped:     {counts['skipped']}",
        ])
    return "\n".join(lines)


def run_annotation(
    root: Path,
    documents: list[Document],
    *,
    edit: bool = False,
    input_fn: Callable[[str], str] = input,
    output: Callable[[str], None] = print,
    viewer: Callable[[Path, Document], Callable[[], None]] = show_document,
) -> int:
    if edit:
        queue = documents
    else:
        queue = sorted(
            (document for document in documents if _status_for(root, document) in {"in_progress", "unannotated"}),
            key=lambda document: (0 if _status_for(root, document) == "in_progress" else 1, DOCUMENT_TYPES.index(document.document_type), document.id.lower()),
        )
    for index, document in enumerate(queue, 1):
        output(f"\nDocument {index} / {len(queue)}")
        output(f"Type: {document.document_type}")
        output("File: " + ", ".join(document.images.values()))
        close = viewer(root, document)
        try:
            result = annotate_one(root, document, edit=edit, input_fn=input_fn, output=output)
        finally:
            close()
        if result == "quit":
            return 0
    if not queue:
        output("No matching unfinished documents.")
    return 0


def validate_dataset(root: Path) -> list[str]:
    issues: list[str] = []
    documents = discover_documents(root, issues)
    by_key = {(document.document_type, document.id): document for document in documents}
    seen_ids: Counter[str] = Counter()
    annotated: set[tuple[str, str]] = set()

    for path in annotation_files(root):
        relative_annotation = path.relative_to(root)
        try:
            data = load_json(path)
        except (OSError, json.JSONDecodeError, ValueError) as error:
            issues.append(f"{relative_annotation}: malformed JSON: {error}")
            continue
        identifier = data.get("id")
        document_type = data.get("document_type")
        if not isinstance(identifier, str) or not identifier:
            issues.append(f"{relative_annotation}: missing valid id")
            continue
        seen_ids[identifier] += 1
        if document_type not in DOCUMENT_TYPES:
            issues.append(f"{relative_annotation}: invalid document_type {document_type!r}")
            continue
        key = (document_type, identifier)
        annotated.add(key)
        document = by_key.get(key)
        if path.stem != identifier or path.parent.name != document_type:
            issues.append(f"{relative_annotation}: path does not match annotation id/type")

        images, hashes = data.get("images"), data.get("image_sha256")
        if not isinstance(images, dict) or set(images) != set(IMAGE_ROLES[document_type]):
            issues.append(f"{relative_annotation}: invalid image roles")
            images = {}
        if not isinstance(hashes, dict) or set(hashes) != set(IMAGE_ROLES[document_type]):
            issues.append(f"{relative_annotation}: invalid image_sha256 roles")
            hashes = {}
        if document and images != document.images:
            issues.append(f"{relative_annotation}: image paths do not match discovered document")
        for role, relative in images.items():
            if not isinstance(relative, str):
                issues.append(f"{relative_annotation}: image path for {role} is not a string")
                continue
            image = (root / relative).resolve()
            if image != root.resolve() and root.resolve() not in image.parents:
                issues.append(f"{relative_annotation}: image path escapes dataset: {relative}")
            elif not image.is_file():
                issues.append(f"{relative_annotation}: missing image: {relative}")
            elif hashes.get(role) != sha256(image):
                issues.append(f"{relative_annotation}: changed SHA-256: {relative}")

        fields = data.get("fields")
        if not isinstance(fields, dict):
            issues.append(f"{relative_annotation}: fields must be an object")
            fields = {}
        unknown = set(fields) - set(FIELDS[document_type])
        if unknown:
            issues.append(f"{relative_annotation}: unknown fields: {', '.join(sorted(unknown))}")
        for name, entry in fields.items():
            if not isinstance(entry, dict) or entry.get("state") not in FIELD_STATES:
                issues.append(f"{relative_annotation}: invalid state for field {name}")
            elif entry["state"] == "value" and not isinstance(entry.get("value"), str):
                issues.append(f"{relative_annotation}: value field {name} must contain a string")
            elif entry["state"] != "value" and entry.get("value") is not None:
                issues.append(f"{relative_annotation}: {entry['state']} field {name} must have null value")

        status = data.get("status")
        if status not in STATUSES:
            issues.append(f"{relative_annotation}: invalid status {status!r}")
        if status != "complete":
            issues.append(f"{relative_annotation}: incomplete annotation ({status})")
        else:
            if "notes" not in data or not isinstance(data.get("notes"), str):
                issues.append(f"{relative_annotation}: missing valid notes")
        missing = set(FIELDS[document_type]) - set(fields)
        if missing:
            issues.append(f"{relative_annotation}: missing annotation fields: {', '.join(sorted(missing))}")

        if document_type in MRZ_SHAPES:
            mrz = data.get("mrz")
            lines = mrz.get("lines", []) if isinstance(mrz, dict) else []
            if not isinstance(lines, list):
                issues.append(f"{relative_annotation}: mrz.lines must be a list")
            else:
                issues.extend(f"{relative_annotation}: {warning}" for warning in mrz_warnings(document_type, lines))
        elif "mrz" in data:
            issues.append(f"{relative_annotation}: driving_license must not contain MRZ")

    issues.extend(f"duplicate logical document id in annotations: {identifier}" for identifier, count in seen_ids.items() if count > 1)
    for key, document in by_key.items():
        if key not in annotated:
            issues.append(f"{document.document_type}/{document.id}: incomplete annotation (unannotated)")
    return issues


def dataset_summary(root: Path, documents: list[Document]) -> str:
    annotations, malformed = saved_annotations(root)
    by_key = {(item.get("document_type"), item.get("id")): item for item in annotations}
    lines = []
    for document_type in DOCUMENT_TYPES:
        selected = [document for document in documents if document.document_type == document_type]
        statuses = Counter(by_key.get((document_type, document.id), {}).get("status", "unannotated") for document in selected)
        field_entries = [
            entry
            for document in selected
            for entry in by_key.get((document_type, document.id), {}).get("fields", {}).values()
            if isinstance(entry, dict)
        ]
        lines.extend([
            document_type.replace("_", " ").title(),
            f"  documents: {len(selected)}",
            f"  complete: {statuses['complete']}",
            f"  incomplete: {statuses['in_progress'] + statuses['unannotated']}",
            f"  unreadable fields: {sum(entry.get('state') == 'unreadable' for entry in field_entries)}",
            f"  skipped: {statuses['skipped']}",
            f"  fields annotated: {len(field_entries)} / {len(selected) * len(FIELDS[document_type])}",
        ])
    if malformed:
        lines.append(f"Malformed annotations: {len(malformed)}")
    return "\n".join(lines)


def export_jsonl(root: Path) -> Path:
    annotations, errors = saved_annotations(root)
    if errors:
        raise ValueError("; ".join(errors))
    path = root / "exports" / "annotations.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
            temporary = handle.name
            for annotation in sorted(annotations, key=lambda item: (str(item.get("document_type")), str(item.get("id")))):
                handle.write(json.dumps(annotation, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("dataset"))
    parser.add_argument("--type", choices=DOCUMENT_TYPES, dest="document_type")
    parser.add_argument("--id", dest="document_id")
    parser.add_argument("--edit", action="store_true", help="edit matching completed/skipped annotations too")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--validate", action="store_true")
    action.add_argument("--summary", action="store_true")
    action.add_argument("--export", action="store_true")
    args = parser.parse_args()
    root = args.dataset.resolve()

    if args.validate:
        issues = validate_dataset(root)
        print(f"Validation: {len(issues)} issue(s)")
        for issue in issues:
            print(f"- {issue}")
        return bool(issues)

    discovery_issues: list[str] = []
    documents = discover_documents(root, discovery_issues)
    if discovery_issues:
        parser.error("; ".join(discovery_issues))
    if args.summary:
        print(dataset_summary(root, documents))
        return 0
    if args.export:
        try:
            path = export_jsonl(root)
        except ValueError as error:
            parser.error(str(error))
        print(f"Exported {path}")
        return 0

    ensure_layout(root)
    documents = discover_documents(root)
    if args.document_type:
        documents = [document for document in documents if document.document_type == args.document_type]
    if args.document_id:
        documents = [document for document in documents if document.id == args.document_id]
        if not documents:
            parser.error(f"document id not found: {args.document_id}")
    print(startup_summary(root, discover_documents(root)))
    return run_annotation(root, documents, edit=args.edit)


if __name__ == "__main__":
    raise SystemExit(main())
