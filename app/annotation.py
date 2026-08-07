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

from app.config import Settings
from app.documents.passport_localization import relative_to_mrz_width
from app.imaging import draw_polygon, order_corners, warp_to_size
from app.models import Models
from app.roi import crop_normalized_roi, draw_roi_assignments, normalized_roi_to_pixels

IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LAYOUTS = ROOT / "config" / "annotation_layouts.json"


@dataclass(frozen=True)
class Sample:
    key: str
    document_type: str
    layout: str
    side: str | None
    path: Path
    annotation_mode: str = "document"
    canonical_size: dict[str, int] | None = None
    seed: dict[str, str] | None = None


def _layouts(layouts_path: Path = DEFAULT_LAYOUTS) -> list[dict[str, Any]]:
    try:
        data = json.loads(layouts_path.read_text(encoding="utf-8"))
        layouts = data["layouts"]
    except (OSError, TypeError, ValueError, KeyError) as error:
        raise ValueError(f"Invalid annotation layout definition {layouts_path}: {error}") from error
    if not isinstance(layouts, list) or not layouts:
        raise ValueError(f"Annotation layout definition has no layouts: {layouts_path}")
    return layouts


def _sample(layout: dict[str, Any], path: Path, key: str, side: str | None = None) -> Sample:
    size = layout.get("canonical_size")
    if size is not None and (not isinstance(size, dict) or not all(isinstance(size.get(name), int) and size[name] > 0 for name in ("width", "height"))):
        raise ValueError(f"Invalid canonical_size for annotation layout {layout.get('layout')}")
    return Sample(key, layout["document_type"], layout["layout"], side, path, layout.get("annotation_mode", "document"), size, layout.get("seed"))


def discover_inputs(input_dir: Path, layouts_path: Path = DEFAULT_LAYOUTS) -> list[Sample]:
    """Discover annotation inputs from the JSON-defined layouts in stable order."""
    samples: list[Sample] = []
    for layout in _layouts(layouts_path):
        directory = input_dir / layout["input_directory"]
        if not directory.exists():
            continue
        if not directory.is_dir():
            raise ValueError(f"Expected directory: {directory}")
        prefix = layout["document_type"]
        sides = layout.get("pair_sides")
        if sides:
            for pair in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
                if not pair.is_dir():
                    raise ValueError(f"{layout['document_type']} inputs must use one directory per pair; found {pair}")
                for side in sides:
                    paths = [item for item in pair.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS and item.stem.lower() == side]
                    if len(paths) != 1:
                        detail = "missing" if not paths else f"duplicate ({', '.join(path.name for path in paths)})"
                        raise ValueError(f"{layout['document_type']} pair {pair.name!r} has {detail} {side} image; use exactly one {side}.<image extension>")
                    samples.append(_sample(layout, paths[0], f"{prefix}:{pair.name}:{side}", side))
        else:
            for path in sorted(directory.iterdir(), key=lambda item: item.name.lower()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    samples.append(_sample(layout, path, f"{prefix}:{path.name}"))
    if not samples:
        raise ValueError(f"No supported images found under {input_dir}; see {layouts_path}")
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


def _rect_points(rect: dict[str, float], width: float, height: float) -> np.ndarray:
    return np.float32([[rect["x1"] * width, rect["y1"] * height], [rect["x2"] * width, rect["y1"] * height], [rect["x2"] * width, rect["y2"] * height], [rect["x1"] * width, rect["y2"] * height]])


def _bounds(points: np.ndarray, width: int, height: int) -> dict[str, float]:
    return {"x1": float(points[:, 0].min() / width), "y1": float(points[:, 1].min() / height), "x2": float(points[:, 0].max() / width), "y2": float(points[:, 1].max() / height)}


def _relative(bounds: dict[str, float], crop: dict[str, float]) -> dict[str, float]:
    return {"x1": (bounds["x1"] - crop["x1"]) / (crop["x2"] - crop["x1"]), "y1": (bounds["y1"] - crop["y1"]) / (crop["y2"] - crop["y1"]), "x2": (bounds["x2"] - crop["x1"]) / (crop["x2"] - crop["x1"]), "y2": (bounds["y2"] - crop["y1"]) / (crop["y2"] - crop["y1"])}


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
                else:
                    self._hydrate(self.data["samples"][sample.key], sample)
                    self._import_legacy_mrz_anchor(self.data["samples"][sample.key])
                    self._migrate_passport_to_mrz_page(self.data["samples"][sample.key])
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
        item = {**asdict(sample), "path": str(sample.path), "original_size": {"width": width, "height": height}, "status": "pending"}
        AnnotationStore._hydrate(item, sample)
        if sample.seed:
            try:
                crop = json.loads((ROOT / sample.seed["data_crop_file"]).read_text(encoding="utf-8"))["data_crop"]
                fields = json.loads((ROOT / sample.seed["fields_file"]).read_text(encoding="utf-8"))
                item.update({"corners": [[0, 0], [1, 0], [1, 1], [0, 1]], "data_crop": crop, "fields": fields, "expected_fields": {}, "mrz": "", "coordinate_space": "canonical", "status": "complete"})
            except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"Cannot seed {sample.key} from annotation layout: {error}") from error
        return item

    @staticmethod
    def _hydrate(item: dict[str, Any], sample: Sample) -> None:
        item.pop("field_names", None)
        item.update({"annotation_mode": sample.annotation_mode, "canonical_size": sample.canonical_size})
        if item.get("annotation_mode") == "mrz_page" and "sec" in item.get("fields", {}):
            item["fields"]["sex"] = item["fields"].pop("sec")
            if "sec" in item.get("expected_fields", {}):
                item["expected_fields"]["sex"] = item["expected_fields"].pop("sec")
        if sample.annotation_mode == "canonical":
            item.setdefault("coordinate_space", "canonical")

    def _import_legacy_mrz_anchor(self, item: dict[str, Any]) -> None:
        """One-time migration: preview JSON is no longer an annotation source."""
        if item.get("annotation_mode") != "mrz_page" or item.get("mrz_anchor"):
            return
        legacy = self.output_dir / "previews" / "passport_mrz.json"
        if not legacy.is_file():
            return
        try:
            anchor = json.loads(legacy.read_text(encoding="utf-8"))
            item["mrz_anchor"] = {
                "mrz_polygon": anchor["mrz_polygon"],
                "page_corners": anchor["page_corners"],
                "page_corners_relative_to_mrz_width": anchor.get("page_corners_relative_to_mrz_width") or relative_to_mrz_width(anchor["mrz_polygon"], anchor["page_corners"]),
            }
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(f"Cannot import legacy passport MRZ anchor: {error}") from error

    @staticmethod
    def _migrate_passport_to_mrz_page(item: dict[str, Any]) -> None:
        """Move old source-canvas ROIs onto the MRZ-anchored rectified page."""
        if item.get("annotation_mode") != "mrz_page" or item.get("coordinate_space") == "canonical" or not item.get("mrz_anchor") or "data_crop" not in item:
            return
        try:
            source_width, source_height = item["original_size"]["width"], item["original_size"]["height"]
            size = item["canonical_size"]
            width, height = int(size["width"]), int(size["height"])
            page = np.float32(item["mrz_anchor"]["page_corners"])
            transform = cv2.getPerspectiveTransform(page, np.float32([[0, 0], [width, 0], [width, height], [0, height]]))
            crop = _bounds(cv2.perspectiveTransform(_rect_points(item["data_crop"], source_width, source_height)[None], transform)[0], width, height)
            fields = {}
            for name, roi in item.get("fields", {}).items():
                absolute = {
                    "x1": item["data_crop"]["x1"] + roi["x1"] * (item["data_crop"]["x2"] - item["data_crop"]["x1"]),
                    "y1": item["data_crop"]["y1"] + roi["y1"] * (item["data_crop"]["y2"] - item["data_crop"]["y1"]),
                    "x2": item["data_crop"]["x1"] + roi["x2"] * (item["data_crop"]["x2"] - item["data_crop"]["x1"]),
                    "y2": item["data_crop"]["y1"] + roi["y2"] * (item["data_crop"]["y2"] - item["data_crop"]["y1"]),
                }
                fields[name] = _relative(_bounds(cv2.perspectiveTransform(_rect_points(absolute, source_width, source_height)[None], transform)[0], width, height), crop)
            item.update({"corners": [[0, 0], [1, 0], [1, 1], [0, 1]], "data_crop": crop, "fields": fields, "coordinate_space": "canonical"})
        except (KeyError, TypeError, ValueError, cv2.error) as error:
            raise ValueError(f"Cannot migrate passport {item.get('key', '<unknown>')} to MRZ page: {error}") from error

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


def _canonical_image(image: np.ndarray, sample: dict[str, Any]) -> np.ndarray:
    """Return the image coordinate space in which this sample's ROIs are drawn."""
    if sample.get("annotation_mode") != "mrz_page" or not sample.get("mrz_anchor"):
        return image
    anchor = sample["mrz_anchor"]
    size = sample.get("canonical_size") or {}
    try:
        return warp_to_size(image, np.asarray(anchor["page_corners"], dtype=np.float32), int(size["width"]), int(size["height"]))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid MRZ anchor for {sample['key']}: {error}") from error


def _detect_mrz_anchor(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Detect the MRZ once, then let the user define the page relative to it."""
    detected = Models(Settings.from_env()).mrz_scanner()(image, do_center_crop=False)
    try:
        polygon = order_corners(np.asarray(detected["mrz_polygon"], dtype=np.float32).reshape(4, 2))
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("MRZ detector did not return a usable MRZ polygon") from error
    page = np.asarray(_click_corners(draw_polygon(image, polygon), normalized=False, title="MRZ detected. Click passport page: top-left, top-right, bottom-right, bottom-left"), dtype=np.float32)
    return polygon, page


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
    if sample.get("mrz_anchor"):
        anchor = sample["mrz_anchor"]
        preview = draw_polygon(draw_polygon(preview, np.asarray(anchor["mrz_polygon"], dtype=np.float32)), np.asarray(anchor["page_corners"], dtype=np.float32))
        atomic_write_json(store.output_dir / "previews" / f"{sample['key'].replace(':', '_')}_mrz.json", anchor)
        atomic_write_json(store.output_dir / "previews" / "passport_mrz.json", anchor)
    preview_path = store.output_dir / "previews" / f"{sample['key'].replace(':', '_')}.jpg"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), preview)
    fields_preview = draw_roi_assignments(crop_normalized_roi(_canonical_image(image, sample), sample["data_crop"]), sample.get("fields", {}), {})
    cv2.imwrite(str(preview_path.with_name(preview_path.stem + "_fields.jpg")), fields_preview)


def _ask(prompt: str, choices: str = "") -> str:
    answer = input(prompt).strip().lower()
    if answer in {"q", "quit"}:
        raise KeyboardInterrupt
    if choices and answer not in choices:
        print(f"Choose one of: {choices}")
        return _ask(prompt, choices)
    return answer


def _click_corners(image: np.ndarray, normalized: bool = True, title: str = "Click corners: top-left, top-right, bottom-right, bottom-left") -> list[list[float]]:
    title, points = f"{title} (u undo, r reset)", []
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
            selected = [[x / width, y / height] for x, y in points]
            return validate_corners(selected) if normalized else order_corners(np.asarray(points, dtype=np.float32)).tolist()
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
    mode = sample.get("annotation_mode", "document")
    if mode == "mrz_page":
        print("Detecting MRZ. The next window is the MRZ overlay; click the passport page, then annotate the rectified page.")
        mrz, page = _detect_mrz_anchor(image)
        sample["mrz_anchor"] = {
            "mrz_polygon": mrz.tolist(),
            "page_corners": page.tolist(),
            "page_corners_relative_to_mrz_width": relative_to_mrz_width(mrz, page),
        }
        sample["corners"] = [[0, 0], [1, 0], [1, 1], [0, 1]]
        sample["coordinate_space"] = "canonical"
    elif mode == "canonical":
        sample["corners"] = [[0, 0], [1, 0], [1, 1], [0, 1]]
        sample["coordinate_space"] = "canonical"
    else:
        sample["corners"] = _click_corners(image)
    store.save()
    image = _canonical_image(image, sample)
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


def run(input_dir: Path, output_dir: Path, check: bool = False, layouts_path: Path = DEFAULT_LAYOUTS) -> int:
    samples = discover_inputs(input_dir, layouts_path)
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
