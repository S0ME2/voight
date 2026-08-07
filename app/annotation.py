"""Local, OpenCV-only annotation workflow for the supplied document layouts."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.imaging import draw_polygon
from app.roi import crop_normalized_roi, draw_roi_assignments, normalized_roi_to_pixels

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Sample:
    key: str
    document_type: str
    layout: str
    side: str | None
    path: Path


def discover_inputs(input_dir: Path) -> list[Sample]:
    """Discover the fixed passport/ID-card input layout in a stable order."""
    samples: list[Sample] = []
    passports = input_dir / "passports"
    if passports.exists():
        if not passports.is_dir():
            raise ValueError(f"Expected directory: {passports}")
        for path in sorted(passports.iterdir(), key=lambda item: item.name.lower()):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append(Sample(f"passport:{path.name}", "passport", "uzbekistan_passport", None, path))
    cards = input_dir / "id_cards"
    if cards.exists():
        if not cards.is_dir():
            raise ValueError(f"Expected directory: {cards}")
        for pair in sorted(cards.iterdir(), key=lambda item: item.name.lower()):
            if not pair.is_dir():
                raise ValueError(f"ID cards must use one directory per pair; found {pair}")
            sides: dict[str, list[Path]] = {"front": [], "back": []}
            for path in pair.iterdir():
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS and path.stem.lower() in sides:
                    sides[path.stem.lower()].append(path)
            for side, paths in sides.items():
                if len(paths) != 1:
                    detail = "missing" if not paths else f"duplicate ({', '.join(path.name for path in paths)})"
                    raise ValueError(f"ID-card pair {pair.name!r} has {detail} {side} image; use exactly one {side}.<image extension>")
                path = paths[0]
                samples.append(Sample(f"id_card:{pair.name}:{side}", "id_card", "uzbekistan_id_card", side, path))
    if not samples:
        raise ValueError(f"No supported images found under {input_dir}; expected passports/ and/or id_cards/")
    return samples


def validate_rect(rect: dict[str, Any]) -> dict[str, float]:
    try:
        checked = {name: float(rect[name]) for name in ("x1", "y1", "x2", "y2")}
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid normalized rectangle: {rect}") from error
    normalized_roi_to_pixels(checked, 100, 100)
    return checked


def validate_corners(corners: list[list[float]] | list[tuple[float, float]]) -> list[list[float]]:
    points = np.asarray(corners, dtype=np.float32)
    if points.shape != (4, 2) or not np.isfinite(points).all() or (points < 0).any() or (points > 1).any():
        raise ValueError("Document corners must be four normalized points between 0 and 1")
    if len({tuple(point) for point in points}) != 4 or not cv2.isContourConvex(points):
        raise ValueError("Document corners must form a non-self-intersecting quadrilateral")
    if abs(cv2.contourArea(points)) < 0.0001:
        raise ValueError("Document corners cover no usable area")
    return points.tolist()


def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        temp_name = handle.name
    os.replace(temp_name, path)


class AnnotationStore:
    def __init__(self, output_dir: Path, samples: list[Sample]):
        self.output_dir = output_dir
        self.path = output_dir / "annotation_state.json"
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
            existing = set(self.data.get("samples", {}))
            wanted = {sample.key for sample in samples}
            removed = existing - wanted
            if removed:
                raise ValueError(f"Input is missing saved sample(s): {', '.join(sorted(removed))}; restore them before resuming")
            for sample in samples:
                if sample.key not in existing:
                    self.data["samples"][sample.key] = self._new_sample(sample)
            if wanted != existing:
                self.save()
        else:
            self.data = {"version": 1, "samples": {sample.key: self._new_sample(sample) for sample in samples}}
            self.save()

    @staticmethod
    def _new_sample(sample: Sample) -> dict[str, Any]:
        image = cv2.imread(str(sample.path))
        if image is None:
            raise ValueError(f"Cannot read image: {sample.path}")
        height, width = image.shape[:2]
        return {**asdict(sample), "path": str(sample.path), "original_size": {"width": width, "height": height}, "status": "pending"}

    def save(self) -> None:
        atomic_write_json(self.path, self.data)

    def sample(self, key: str) -> dict[str, Any]:
        return self.data["samples"][key]


def profile_key(sample: dict[str, Any]) -> str:
    return sample["layout"] + (f"_{sample['side']}" if sample["side"] else "")


def validate_state(data: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    profiles: dict[str, Any] = {}
    for key, sample in data.get("samples", {}).items():
        if sample.get("status") == "skipped":
            continue
        try:
            validate_corners(sample["corners"])
            validate_rect(sample["data_crop"])
            fields = sample.get("fields", {})
            if not isinstance(fields, dict):
                raise ValueError("fields must be an object")
            for name, rect in fields.items():
                if not name or not isinstance(name, str):
                    raise ValueError("field names must be non-empty strings")
                validate_rect(rect)
            profile = {"layout": sample["layout"], "side": sample["side"], "data_crop": sample["data_crop"], "field_rois": fields}
            profiles.setdefault(profile_key(sample), profile)
        except (KeyError, TypeError, ValueError) as error:
            errors.append(f"{key}: {error}")
    if not data.get("samples"):
        errors.append("No samples in annotation state")
    return errors


def write_outputs(store: AnnotationStore) -> dict[str, Any]:
    errors = validate_state(store.data)
    samples = store.data["samples"]
    profiles: dict[str, Any] = {}
    truth: dict[str, Any] = {}
    for key, sample in samples.items():
        if sample.get("status") == "skipped":
            continue
        if sample.get("status") != "complete":
            continue
        try:
            validate_corners(sample["corners"])
            validate_rect(sample["data_crop"])
            for rect in sample.get("fields", {}).values():
                validate_rect(rect)
        except (KeyError, TypeError, ValueError):
            continue
        profile = {"layout": sample["layout"], "side": sample["side"], "data_crop": sample["data_crop"], "field_rois": sample.get("fields", {})}
        profiles.setdefault(profile_key(sample), profile)
        truth[key] = {"source": sample["path"], "original_size": sample["original_size"], "corners": sample["corners"], "expected_fields": sample.get("expected_fields", {}), "mrz": sample.get("mrz", "")}
    for name, profile in profiles.items():
        atomic_write_json(store.output_dir / "profiles" / f"{name}.json", profile)
    atomic_write_json(store.output_dir / "evaluation_ground_truth.json", {"samples": truth})
    report = {"valid": not errors, "errors": errors, "samples": {status: sum(item.get("status") == status for item in samples.values()) for status in ("pending", "complete", "skipped")}, "profiles": sorted(profiles)}
    atomic_write_json(store.output_dir / "progress.json", report)
    return report


def write_preview(store: AnnotationStore, sample: dict[str, Any]) -> None:
    image = cv2.imread(sample["path"])
    if image is None:
        raise ValueError(f"Cannot read image: {sample['path']}")
    height, width = image.shape[:2]
    corners = np.asarray(sample["corners"], dtype=np.float32) * np.array([width, height], dtype=np.float32)
    preview = draw_polygon(image, corners)
    preview_path = store.output_dir / "previews" / f"{sample['key'].replace(':', '_')}.jpg"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), preview)
    fields_preview = draw_roi_assignments(crop_normalized_roi(image, sample["data_crop"]), sample.get("fields", {}), {})
    cv2.imwrite(str(preview_path.with_name(preview_path.stem + "_fields.jpg")), fields_preview)


def _ask(prompt: str, choices: str = "") -> str:
    answer = input(prompt).strip().lower()
    if answer in {"q", "quit"}:
        raise KeyboardInterrupt
    if choices and answer not in choices:
        print(f"Choose one of: {choices}")
        return _ask(prompt, choices)
    return answer


def _click_corners(image: np.ndarray) -> list[list[float]]:
    title, points = "Click corners: top-left, top-right, bottom-right, bottom-left (u undo, r reset)", []
    shown = image.copy()

    def click(event: int, x: int, y: int, _flags: int, _data: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
            points.append((x, y))

    cv2.namedWindow(title)
    cv2.setMouseCallback(title, click)
    while True:
        frame = shown.copy()
        for index, point in enumerate(points, 1):
            cv2.circle(frame, point, 6, (0, 255, 0), -1)
            cv2.putText(frame, str(index), point, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow(title, frame)
        key = cv2.waitKey(20) & 0xFF
        if key == ord("u") and points:
            points.pop()
        elif key == ord("r"):
            points.clear()
        elif key in (13, 32) and len(points) == 4:
            cv2.destroyWindow(title)
            height, width = image.shape[:2]
            return validate_corners([[x / width, y / height] for x, y in points])
        elif key in (27, ord("q")):
            cv2.destroyWindow(title)
            raise KeyboardInterrupt


def _select_rect(image: np.ndarray, title: str) -> dict[str, float] | None:
    x, y, width, height = cv2.selectROI(title, image, showCrosshair=True, fromCenter=False)
    cv2.destroyWindow(title)
    if not width or not height:
        return None
    image_height, image_width = image.shape[:2]
    return validate_rect({"x1": x / image_width, "y1": y / image_height, "x2": (x + width) / image_width, "y2": (y + height) / image_height})


def annotate_sample(store: AnnotationStore, sample: dict[str, Any]) -> None:
    image = cv2.imread(sample["path"])
    if image is None:
        raise ValueError(f"Cannot read image: {sample['path']}")
    print(f"\n{sample['key']}: confirm {sample['document_type']} / {sample['layout']}" + (f" / {sample['side']}" if sample["side"] else ""))
    action = _ask("Enter to annotate; [s]kip, [b]ack to menu, [q]uit: ")
    if action == "s":
        sample["status"] = "skipped"
        store.save()
        return
    if action == "b":
        return
    sample["corners"] = _click_corners(image)
    store.save()
    canonical = next((item for item in store.data["samples"].values() if item is not sample and item.get("status") == "complete" and profile_key(item) == profile_key(sample) and "data_crop" in item), None)
    if canonical and _ask(f"Reuse field geometry from {canonical['key']}? Enter=yes, r=redraw: ") != "r":
        sample["data_crop"] = canonical["data_crop"]
        fields = canonical.get("fields", {}).copy()
    else:
        while True:
            crop = _select_rect(image, "Draw canonical data/text crop; Enter accepts, c cancels")
            if crop:
                sample["data_crop"] = crop
                store.save()
                break
            print("A data/text crop is required. Draw a rectangle and press Enter.")
        fields = {}
        field_image = crop_normalized_roi(image, sample["data_crop"])
        print("Draw each field rectangle in the data/text crop. Cancel selection when finished; type u to remove the last field.")
        while True:
            rect = _select_rect(field_image, "Draw field ROI in data/text crop; Enter accepts, c finishes")
            if rect is None:
                command = _ask("Field name, [u]ndo last, or Enter to finish: ")
                if command == "u" and fields:
                    fields.pop(next(reversed(fields)))
                    store.save()
                elif not command:
                    break
                else:
                    print("Draw a rectangle before entering its field name.")
                continue
            name = _ask("Field name ([u]ndo selection, [q]uit): ")
            if name == "u":
                continue
            if not name:
                print("Field name is required.")
                continue
            fields[name] = rect
            sample["fields"] = fields
            store.save()
    sample["fields"] = fields
    expected = {name: input(f"Expected value for {name} (blank allowed): ").strip() for name in fields}
    sample["expected_fields"] = expected
    sample["mrz"] = input("Expected MRZ text (blank if absent; use < exactly): ").strip()
    sample["status"] = "complete"
    store.save()
    write_preview(store, sample)


def run(input_dir: Path, output_dir: Path, check: bool = False) -> int:
    samples = discover_inputs(input_dir)
    store = AnnotationStore(output_dir, samples)
    if check:
        report = write_outputs(store)
        print(json.dumps(report, indent=2))
        return 0 if report["valid"] else 1
    try:
        while True:
            pending = next((store.sample(item.key) for item in samples if store.sample(item.key).get("status") == "pending"), None)
            if pending is None:
                completed = [item for item in samples if store.sample(item.key).get("status") == "complete"]
                if not completed:
                    break
                choices = ", ".join(f"{index + 1}:{item.key}" for index, item in enumerate(completed))
                edit = _ask(f"All inputs handled. Enter to finish, or type completed sample number to edit ({choices}): ")
                if not edit:
                    break
                if edit.isdigit() and 1 <= int(edit) <= len(completed):
                    sample = store.sample(completed[int(edit) - 1].key)
                    sample["status"] = "pending"
                    store.save()
                    continue
                print("Enter a listed number or press Enter to finish.")
                continue
            annotate_sample(store, pending)
            store.save()
    except KeyboardInterrupt:
        print("Saved. Resume with the same command.")
        return 0
    report = write_outputs(store)
    print(f"Saved profiles, evaluation_ground_truth.json, previews, and progress.json under {output_dir}.")
    return 0 if report["valid"] else 1
