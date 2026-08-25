"""Measure CPU preprocessing on fixed Voight recognition crops and the real API."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.artifacts import ArtifactWriter
from app.config import Settings
from app.documents.mrz import MrzProfile, crop_polygon, parse as parse_mrz, preprocess as current_mrz_preprocess
from app.documents.passport_localization import page_corners_from_mrz_width
from app.documents.profiles import load_document_profile
from app.imaging import preprocess_variant, warp_to_size
from app.inference.batch import _line_crop
from app.inference.packing import recognition_batch_packer
from app.models import Models
from app.pipeline import RegionProfile, prepare_profile_from_detection
from app.roi import normalized_roi_to_pixels, roi_for_point
from benchmarks.maintained.batch_size_benchmark import _differences, _snapshot
from benchmarks.maintained.model_matrix_benchmark import Server, _post, _score
from benchmarks.maintained.pipeline_breakdown import (
    DOC_TYPES,
    annotation_truth,
    discover_dataset,
    validate_and_manifest,
)

VISIBLE_VARIANTS = (
    "original", "grayscale", "contrast_1.15", "contrast_1.30", "contrast_1.50",
    "clahe_mild", "clahe_medium", "gamma_0.8", "gamma_1.2", "sharpen_light",
)
MRZ_VARIANTS = VISIBLE_VARIANTS + ("otsu", "adaptive")
BASE_ENV = {
    "RUNTIME_TARGET": "cpu", "OCR_DEVICE": "cpu", "PRELOAD": "true", "LOGGING": "false",
    "CPU_THREADS": "4", "LOCALIZATION_BATCH_SIZE": "4", "TEXT_DETECTION_BATCH_SIZE": "1",
    "TEXT_RECOGNITION_BATCH_SIZE": "2", "MRZ_RECOGNITION_BATCH_SIZE": "2",
    "TEXT_RECOGNITION_PROCESSES": "1", "TEXT_RECOGNITION_PACKING": "fixed-width",
    "TEXT_DETECTOR_MODEL": "PP-OCRv6_medium_det", "TEXT_RECOGNIZER_MODEL": "latin_PP-OCRv5_mobile_rec",
    "DOCALIGNER_MODEL": "fastvit_sa24", "DOCALIGNER_MODEL_TYPE": "heatmap",
    "MRZ_RECOGNIZER_BACKEND": "generic-paddle", "MRZ_RECOGNIZER_MODEL": "20250221",
    "OCR_MAX_SIDE": "3000", "OCR_CONTRAST": "1.25", "MRZ_POLYGON_PADDING_RATIO": "0.03",
    "DOCALIGNER_PADDING": "100", "TEXT_DETECTOR_LIMIT_SIDE_LEN": "960",
}


@dataclass
class Crop:
    key: str
    document_id: str
    document_type: str
    role: str
    field: str | None
    expected: str | None
    image: np.ndarray


@dataclass
class DetectorInput:
    key: str
    document_id: str
    document_type: str
    role: str
    image: np.ndarray
    rois: dict[str, dict[str, float]] | None


def _distance(left: str, right: str) -> int:
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        next_row = [i]
        for j, b in enumerate(right, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = next_row
    return row[-1]


def _norm(value: Any) -> str:
    return "".join(str(value or "").upper().split())


def _sha(image: np.ndarray) -> str:
    return hashlib.sha256(image.tobytes()).hexdigest()


def _truth(document: Any) -> dict[str, Any]:
    return annotation_truth(document)


def _read(document: Any) -> dict[str, np.ndarray]:
    values = {}
    for role, path in document.paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode {path}")
        values[role] = image
    return values


def _localize(localizer: Any, jobs: list[tuple[str, np.ndarray]], batch: int, pad: int = 0) -> dict[str, Any]:
    result = {}
    for start in range(0, len(jobs), batch):
        chunk = jobs[start:start + batch]
        images = [cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT) if pad else image for _, image in chunk]
        result.update(zip((key for key, _ in chunk), localizer.localize_batch(images)))
    return result


def _profile(settings: Settings, kind: str, role: str) -> tuple[RegionProfile, int, int]:
    if kind == "driving_license":
        from app.pipeline import load_region_profile
        return load_region_profile(settings.driving_license.data_crop, settings.driving_license.field_rois), settings.driving_license.canonical_width, settings.driving_license.canonical_height
    profile = load_document_profile(settings.profiles.passport if kind == "passport" else settings.profiles.id_card)
    region = "data_page" if kind == "passport" else role
    value = profile["regions"][region]
    return RegionProfile(value["data_crop"], value["field_rois"]), int(profile["canonical_size"]["width"]), int(profile["canonical_size"]["height"])


def build_fixed_crops(settings: Settings, documents: list[Any], models: Models, output: Path) -> tuple[list[Crop], list[DetectorInput], dict[str, Any]]:
    artifact = ArtifactWriter(output, "", "unused", False)
    images = {document.document_id: _read(document) for document in documents}
    doc_jobs = [(f"{document.document_id}:{role}", image) for document in documents for role, image in images[document.document_id].items()]
    docaligner = models.document_localizer()
    docaligned = _localize(docaligner, doc_jobs, 4, settings.driving_license.aligner_padding)
    passport_jobs = [(f"{document.document_id}:image", images[document.document_id]["image"]) for document in documents if document.document_type == "passport"]
    mrz_jobs = [(f"{document.document_id}:back", images[document.document_id]["back"]) for document in documents if document.document_type == "id_card"]
    mrz_jobs += passport_jobs
    mrz_locations = _localize(models.mrz_localizer(), mrz_jobs, 4)

    detector_inputs: list[DetectorInput] = []
    for document in documents:
        truth = _truth(document)
        for role, image in images[document.document_id].items():
            profile, width, height = _profile(settings, document.document_type, role)
            if document.document_type == "passport":
                polygon = mrz_locations[f"{document.document_id}:image"].polygon.reshape(4, 2)
                page = load_document_profile(settings.profiles.passport)["document_localization"]["page_corners_relative_to_mrz_width"]
                corners = page_corners_from_mrz_width(polygon, page)
                prepared = prepare_profile_from_detection(image, profile, corners, artifact, canonical_width=width, canonical_height=height, padding=0, padded_corners=corners)
                detector_inputs.append(DetectorInput(f"visible:{document.document_id}:{role}", document.document_id, document.document_type, "visible", prepared.data_crop, profile.field_rois))
            elif document.document_type == "id_card":
                padded_corners = docaligned[f"{document.document_id}:{role}"].polygon.reshape(4, 2)
                corners = padded_corners - settings.driving_license.aligner_padding
                prepared = prepare_profile_from_detection(image, profile, corners, artifact, canonical_width=width, canonical_height=height, padding=settings.driving_license.aligner_padding, padded_corners=padded_corners)
                detector_inputs.append(DetectorInput(f"visible:{document.document_id}:{role}", document.document_id, document.document_type, "visible", prepared.data_crop, profile.field_rois))
            else:
                padded_corners = docaligned[f"{document.document_id}:{role}"].polygon.reshape(4, 2)
                corners = padded_corners - settings.driving_license.aligner_padding
                prepared = prepare_profile_from_detection(image, profile, corners, artifact, canonical_width=width, canonical_height=height, padding=settings.driving_license.aligner_padding, padded_corners=padded_corners)
                detector_inputs.append(DetectorInput(f"visible:{document.document_id}:{role}", document.document_id, document.document_type, "visible", prepared.data_crop, profile.field_rois))

        if document.document_type == "passport":
            role, source = "image", images[document.document_id]["image"]
        elif document.document_type == "id_card":
            role, source = "back", images[document.document_id]["back"]
        else:
            continue
        polygon = mrz_locations[f"{document.document_id}:{role}"].polygon.reshape(4, 2)
        crop, _ = crop_polygon(source, polygon, settings.mrz.polygon_padding_ratio)
        # This is the current MRZ normalization used before text detection.
        normalized = current_mrz_preprocess(crop, settings.mrz.max_side, settings.mrz.contrast)
        detector_inputs.append(DetectorInput(f"mrz:{document.document_id}:{role}", document.document_id, document.document_type, "mrz", normalized, None))

    detector = models.text_detector()
    crops: list[Crop] = []
    baseline_boxes: dict[str, list[list[float]]] = {}
    for source in detector_inputs:
        result = detector.detect_batch([source.image])[0]
        boxes = [np.asarray(region.polygon, dtype=np.float32) for region in result.regions]
        boxes.sort(key=lambda polygon: (float(polygon[:, 1].mean()), float(polygon[:, 0].min())))
        baseline_boxes[source.key] = [polygon.tolist() for polygon in boxes]
        truth = next(document for document in documents if document.document_id == source.document_id)
        expected_lines = _truth(truth).get("mrz", {}).get("lines", [])
        for index, polygon in enumerate(boxes):
            if source.role == "visible":
                center = (polygon.min(axis=0) + polygon.max(axis=0)) / 2
                field = roi_for_point(source.rois or {}, source.image.shape[1], source.image.shape[0], float(center[0]), float(center[1]))
                if field is None:
                    continue
                expected = _truth(truth).get("fields", {}).get(field, {}).get("value")
            else:
                field = f"line_{index + 1}"
                expected = expected_lines[index] if index < len(expected_lines) else None
            line, _ = _line_crop(source.image, polygon)
            crops.append(Crop(f"{source.key}:{index}", source.document_id, source.document_type, source.role, field, expected, line))

    mrz_source_hashes = {
        document.document_id: _sha(
            images[document.document_id]["image" if document.document_type == "passport" else "back"]
        )
        for document in documents
        if document.document_type in {"passport", "id_card"}
    }
    manifest = {
        "schema_version": 1,
        "source": "current pipeline localization/canonicalization and detector once; crops then frozen",
        "settings": {key: os.environ.get(key) for key in BASE_ENV},
        "detector_inputs": [{"key": item.key, "document_id": item.document_id, "document_type": item.document_type, "role": item.role, "shape": list(item.image.shape), "sha256": _sha(item.image), **({"source_image_sha256": mrz_source_hashes[item.document_id]} if item.role == "mrz" else {})} for item in detector_inputs],
        "recognition_crops": [{"key": crop.key, "document_id": crop.document_id, "document_type": crop.document_type, "role": crop.role, "field": crop.field, "expected": crop.expected, "shape": list(crop.image.shape), "sha256": _sha(crop.image), **({"source_image_sha256": mrz_source_hashes[crop.document_id]} if crop.role == "mrz" else {})} for crop in crops],
        "baseline_boxes": baseline_boxes,
    }
    (output / "fixed_crops").mkdir(parents=True, exist_ok=True)
    (output / "fixed_crops" / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    for crop in crops:
        np.save(output / "fixed_crops" / f"{crop.key.replace(':', '__')}.npy", crop.image)
    for item in detector_inputs:
        np.save(output / "fixed_crops" / f"input__{item.key.replace(':', '__')}.npy", item.image)
    return crops, detector_inputs, manifest


def recognize(crops: list[Crop], recognizer: Any, variant: str, batch_size: int, repeats: int, warmup: int = 1) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected = [crop for crop in crops if crop.role == "visible" or crop.role == "mrz"]
    packer = recognition_batch_packer("fixed-width")
    def one() -> tuple[dict[str, Any], dict[str, float], dict[str, Any]]:
        started = time.perf_counter()
        transformed = []
        preprocess_seconds = 0.0
        for crop in selected:
            tick = time.perf_counter(); transformed.append(preprocess_variant(crop.image, variant)); preprocess_seconds += time.perf_counter() - tick
        outputs: dict[int, Any] = {}
        calls = []
        recognition_started = time.perf_counter()
        for chunk in packer.pack(list(enumerate(transformed)), batch_size):
            values = recognizer.recognize_batch([image for _, image in chunk])
            outputs.update((index, value) for (index, _), value in zip(chunk, values))
            calls.append({"submitted_batch_size": len(chunk), "tensor_batch_sizes": list(getattr(recognizer, "last_tensor_batch_sizes", [len(chunk)]))})
        recognition_seconds = time.perf_counter() - recognition_started
        records = []
        for index, crop in enumerate(selected):
            text = str(outputs[index].text).strip()
            records.append({"key": crop.key, "document_id": crop.document_id, "document_type": crop.document_type, "role": crop.role, "field": crop.field, "expected": crop.expected, "actual": text, "score": outputs[index].score})
        return {"records": records}, {"preprocessing_seconds": preprocess_seconds, "recognition_seconds": recognition_seconds, "total_crop_processing_seconds": time.perf_counter() - started}, {"calls": calls, "crop_count": len(selected)}
    for _ in range(warmup):
        one()
    rows = []
    for repeat in range(1, repeats + 1):
        value, timings, batches = one()
        rows.append({"repeat": repeat, "variant": variant, **timings, **summarize_recognition([{"outputs": value["records"]}]), "batches": batches, "outputs": value["records"]})
    return rows, summarize_recognition(rows)


def summarize_recognition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    records = rows[-1]["outputs"]
    fields = {doc: {} for doc in {row["document_id"] for row in records}}
    mrz = {doc: {} for doc in fields}
    for row in records:
        target = mrz if row["role"] == "mrz" else fields
        target[row["document_id"]].setdefault(row["field"], []).append(row["actual"])
    field_exact = field_total = field_chars = field_char_total = 0
    by_kind = {kind: {"exact": 0, "total": 0, "characters": 0, "character_total": 0} for kind in DOC_TYPES}
    for row in records:
        if row["role"] != "visible" or row["field"] is None or row["expected"] is None:
            continue
        actual = _norm(row["actual"]); expected = _norm(row["expected"]); kind = row["document_type"]
        target = by_kind[kind]; target["total"] += 1; target["character_total"] += max(1, len(expected)); target["characters"] += max(1, len(expected)) - _distance(expected, actual); target["exact"] += actual == expected
        field_total += 1; field_char_total += max(1, len(expected)); field_chars += max(1, len(expected)) - _distance(expected, actual); field_exact += actual == expected
    by_doc_mrz = {}
    for doc, values in mrz.items():
        lines = [values[key][0] for key in sorted(values)]
        by_doc_mrz[doc] = lines
    mrz_docs = {doc: lines for doc, lines in by_doc_mrz.items() if any(row["role"] == "mrz" and row["document_id"] == doc for row in records)}
    mrz_exact = mrz_line_exact = mrz_chars = mrz_char_total = 0
    mrz_valid = 0
    for doc, actual_lines in mrz_docs.items():
        expected_lines = [row["expected"] for row in records if row["role"] == "mrz" and row["document_id"] == doc and row["expected"] is not None]
        expected_lines = expected_lines[:len(actual_lines)]
        mrz_exact += actual_lines == expected_lines and len(actual_lines) == len(expected_lines)
        for expected, actual in zip(expected_lines, actual_lines):
            expected, actual = _norm(expected), _norm(actual); mrz_line_exact += expected == actual; mrz_char_total += len(expected); mrz_chars += len(expected) - _distance(expected, actual)
        parsed = parse_mrz("\n".join(actual_lines), "passport" if doc.startswith("p_") else "id_card")
        mrz_valid += bool(parsed.raw_lines) and all(item.status.value != "failed" for item in parsed.validations)
    return {
        "field_exact": field_exact, "field_total": field_total, "field_exact_rate": field_exact / field_total if field_total else None,
        "field_characters_correct": field_chars, "field_character_total": field_char_total, "field_character_errors": field_char_total - field_chars, "field_character_accuracy": field_chars / field_char_total if field_char_total else None,
        "by_document_type": by_kind, "mrz_documents": len(mrz_docs), "mrz_exact_documents": mrz_exact, "mrz_exact_rate": mrz_exact / len(mrz_docs) if mrz_docs else None,
        "mrz_line_exact": mrz_line_exact, "mrz_line_total": sum(1 for row in records if row["role"] == "mrz" and row["expected"] is not None), "mrz_character_errors": mrz_char_total - mrz_chars, "mrz_character_accuracy": mrz_chars / mrz_char_total if mrz_char_total else None, "mrz_check_digit_valid": mrz_valid,
    }


def _detector_lines(detector: Any, item: DetectorInput, variant: str) -> tuple[list[dict[str, Any]], float, float]:
    tick = time.perf_counter(); transformed = preprocess_variant(item.image, variant); preprocessing = time.perf_counter() - tick
    tick = time.perf_counter(); result = detector.detect_batch([transformed])[0]; detection = time.perf_counter() - tick
    boxes = [np.asarray(region.polygon, dtype=np.float32) for region in result.regions]
    boxes.sort(key=lambda polygon: (float(polygon[:, 1].mean()), float(polygon[:, 0].min())))
    return [{"polygon": box, "crop": _line_crop(item.image, box)[0]} for box in boxes], preprocessing, detection


def _box_match(left: np.ndarray, right: np.ndarray) -> bool:
    lc = (left.min(axis=0) + left.max(axis=0)) / 2; rc = (right.min(axis=0) + right.max(axis=0)) / 2
    return float(np.linalg.norm(lc - rc)) <= max(5.0, float(max(np.ptp(left[:, 1]), np.ptp(right[:, 1])) * 1.5))


def run_detector_phase(documents: list[Any], inputs: list[DetectorInput], detector: Any, recognizer: Any, crops: list[Crop], repeats: int, output: Path) -> list[dict[str, Any]]:
    baseline = {item.key: [np.asarray(value) for value in json.loads((output / "fixed_crops" / "manifest.json").read_text())["baseline_boxes"].get(item.key, [])] for item in inputs}
    rows = []
    baseline_outputs = None
    for variant in ("original", "grayscale", "contrast_1.25", "clahe_mild", "gamma_0.8", "gamma_1.2"):
        for repeat in range(1, repeats + 1):
            began = time.perf_counter(); detected = {}; prep = det = 0.0; missing = extra = 0
            dynamic_crops = []
            for item in inputs:
                lines, p, d = _detector_lines(detector, item, variant); prep += p; det += d; detected[item.key] = lines
                reference = baseline[item.key]; used = set()
                for box in reference:
                    match = next((index for index, candidate in enumerate(lines) if index not in used and _box_match(box, candidate["polygon"])), None)
                    if match is None: missing += 1
                    else: used.add(match)
                extra += len(lines) - len(used)
                for index, line in enumerate(lines):
                    if item.role == "visible":
                        center = (line["polygon"].min(axis=0) + line["polygon"].max(axis=0)) / 2
                        field = roi_for_point(item.rois or {}, item.image.shape[1], item.image.shape[0], float(center[0]), float(center[1]))
                        if field is None: continue
                        expected = _truth(next(doc for doc in documents if doc.document_id == item.document_id)).get("fields", {}).get(field, {}).get("value")
                    else:
                        field = f"line_{index + 1}"; expected_lines = _truth(next(doc for doc in documents if doc.document_id == item.document_id)).get("mrz", {}).get("lines", []); expected = expected_lines[index] if index < len(expected_lines) else None
                    dynamic_crops.append(Crop(f"{item.key}:{index}", item.document_id, item.document_type, item.role, field, expected, line["crop"]))
            recognition_rows, recognition_summary = recognize(dynamic_crops, recognizer, "original", 2, 1, 0)
            row = {"variant": variant, "repeat": repeat, "preprocessing_seconds": prep, "detection_seconds": det, "recognition_preprocessing_seconds": recognition_rows[0]["preprocessing_seconds"], "recognition_seconds": recognition_rows[0]["recognition_seconds"], "e2e_seconds": time.perf_counter() - began, "detected_lines": sum(len(value) for value in detected.values()), "missing_lines": missing, "extra_lines": extra, **recognition_summary}
            row["outputs"] = recognition_rows[0]["outputs"]
            if baseline_outputs is None and variant == "original": baseline_outputs = row["outputs"]
            row["output_differences"] = _output_difference(baseline_outputs or [], row["outputs"])
            rows.append(row)
    return rows


def _output_difference(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> int:
    left_map = {row["key"]: row["actual"] for row in left}; right_map = {row["key"]: row["actual"] for row in right}
    return sum(left_map.get(key) != right_map.get(key) for key in set(left_map) | set(right_map))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: return
    keys = sorted({key for row in rows for key, value in row.items() if not isinstance(value, (dict, list))})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows({key: row.get(key) for key in keys} for row in rows)


def load_fixed_crops(settings: Settings, documents: list[Any], source: Path) -> tuple[list[Crop], list[DetectorInput], dict[str, Any]]:
    fixed = source / "fixed_crops"
    manifest = json.loads((fixed / "manifest.json").read_text(encoding="utf-8"))
    rois = {}
    for kind, profile_path in (("passport", settings.profiles.passport), ("id_card", settings.profiles.id_card)):
        profile = load_document_profile(profile_path)
        rois[kind] = {region: value["field_rois"] for region, value in profile["regions"].items()}
    from app.pipeline import load_region_profile
    rois["driving_license"] = {"image": load_region_profile(settings.driving_license.data_crop, settings.driving_license.field_rois).field_rois}
    inputs = []
    for row in manifest["detector_inputs"]:
        side = row["key"].split(":")[-1]
        region = "data_page" if row["document_type"] == "passport" else side
        values = rois.get(row["document_type"], {})
        inputs.append(DetectorInput(row["key"], row["document_id"], row["document_type"], row["role"], np.load(fixed / f"input__{row['key'].replace(':', '__')}.npy"), values.get(region) if row["role"] == "visible" else None))
    crops = [Crop(row["key"], row["document_id"], row["document_type"], row["role"], row["field"], row["expected"], np.load(fixed / f"{row['key'].replace(':', '__')}.npy")) for row in manifest["recognition_crops"]]
    return crops, inputs, manifest


def _median_rows(rows: list[dict[str, Any]], phase: str, variant: str) -> dict[str, Any]:
    selected = [row for row in rows if row["variant"] == variant]
    result = {"phase": phase, "variant": variant, "repeats": len(selected)}
    for key in ("preprocessing_seconds", "recognition_preprocessing_seconds", "recognition_seconds", "total_crop_processing_seconds", "detection_seconds", "e2e_seconds", "missing_lines", "extra_lines", "field_exact_rate", "field_character_accuracy", "field_character_errors", "mrz_exact_rate", "mrz_character_accuracy", "mrz_character_errors", "mrz_check_digit_valid", "output_differences"):
        values = [row[key] for row in selected if row.get(key) is not None]
        result[key] = statistics.median(values) if values else None
    return result


def write_by_document_tables(output: Path, recognition_rows: list[dict[str, Any]], detector_rows: list[dict[str, Any]], mrz_rows: list[dict[str, Any]]) -> None:
    def expand(rows: list[dict[str, Any]], phase: str, variants: tuple[str, ...], role: str | None = None) -> list[dict[str, Any]]:
        result = []
        for kind in DOC_TYPES:
            for variant in variants:
                selected = [row for row in rows if row["variant"] == variant]
                scoped = []
                for row in selected:
                    records = [item for item in row["outputs"] if item["document_type"] == kind and (role is None or item["role"] == role)]
                    if records:
                        scoped.append({"outputs": records, **{key: row.get(key) for key in ("preprocessing_seconds", "recognition_seconds", "total_crop_processing_seconds", "detection_seconds", "e2e_seconds", "missing_lines", "extra_lines", "output_differences")}})
                if not scoped:
                    continue
                metric = summarize_recognition(scoped)
                row_out = {"phase": phase, "document_type": kind, "variant": variant, "repeats": len(scoped), **{key: metric.get(key) for key in ("field_exact_rate", "field_character_accuracy", "field_character_errors", "mrz_exact_rate", "mrz_character_accuracy", "mrz_character_errors", "mrz_check_digit_valid", "mrz_line_exact", "mrz_line_total")}}
                for key in ("preprocessing_seconds", "recognition_seconds", "total_crop_processing_seconds", "detection_seconds", "e2e_seconds", "missing_lines", "extra_lines", "output_differences"):
                    values = [item[key] for item in scoped if item.get(key) is not None]
                    row_out[key] = statistics.median(values) if values else None
                result.append(row_out)
        return result
    write_csv(output / "recognition_preprocessing_by_document.csv", expand(recognition_rows, "A", VISIBLE_VARIANTS))
    write_csv(output / "detector_preprocessing_by_document.csv", expand(detector_rows, "B", ("original", "grayscale", "contrast_1.25", "clahe_mild", "gamma_0.8", "gamma_1.2")))
    write_csv(output / "mrz_preprocessing_by_document.csv", expand(mrz_rows, "C", MRZ_VARIANTS, "mrz"))


def write_full_pipeline_report(output: Path, documents: list[Any]) -> None:
    raw = [json.loads(line) for line in (output / "full_pipeline_raw.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    rows = []
    for candidate in sorted({row["candidate"] for row in raw}):
        for kind in DOC_TYPES:
            selected = [row for row in raw if row["candidate"] == candidate and row["document_type"] == kind]
            if not selected: continue
            score = selected[0]["correctness"]
            rows.append({"candidate": candidate, "document_type": kind, "repeats": len(selected), "e2e_seconds_median": statistics.median(row["client_e2e_seconds"] for row in selected), "docs_per_second_median": statistics.median(row["docs_per_second"] for row in selected), "rss_mb_max": max(row.get("rss_mb") or 0 for row in selected), "field_exact_rate": statistics.median(row["correctness"]["field_correctness"] or 0 for row in selected), "field_character_accuracy": statistics.median(row["correctness"]["field_character_accuracy"] or 0 for row in selected), "mrz_exact_rate": statistics.median(row["correctness"]["mrz_exact_match_rate"] or 0 for row in selected), "mrz_character_accuracy": statistics.median(row["correctness"]["mrz_character_accuracy"] or 0 for row in selected), **{f"stage_{name}_seconds": statistics.median(row["stage_timings"].get(name, 0) for row in selected) for name in ("localization", "pipeline", "text_detection", "text_recognition", "mrz_recognition")}})
    write_csv(output / "full_pipeline_finalists.csv", rows)
    finalists = json.loads((output / "finalists.json").read_text(encoding="utf-8"))["candidates"]
    baseline_dir = output / "full_pipeline" / "00_baseline"
    differences = {}
    for index, candidate in enumerate(finalists):
        candidate_dir = output / "full_pipeline" / f"{index:02d}_{candidate['name']}"
        for kind in DOC_TYPES:
            base_path = baseline_dir / f"{kind}_1.json"; candidate_path = candidate_dir / f"{kind}_1.json"
            if not base_path.is_file() or not candidate_path.is_file(): continue
            base = json.loads(base_path.read_text(encoding="utf-8")); value = json.loads(candidate_path.read_text(encoding="utf-8"))
            selected = [document for document in documents if document.document_type == kind]
            differences[f"{candidate['name']}:{kind}"] = _differences(selected, _snapshot(selected, base), _snapshot(selected, value))
    (output / "full_pipeline_output_differences.json").write_text(json.dumps(differences, indent=2), encoding="utf-8")


def write_fixed_output_differences(output: Path, filename: str, target: str) -> None:
    rows = [json.loads(line) for line in (output / filename).read_text(encoding="utf-8").splitlines() if line.strip()]
    baseline = next(row["outputs"] for row in rows if row["variant"] == "original")
    def values(records: list[dict[str, Any]]) -> dict[tuple[str, str, str], str]:
        return {(row["document_id"], row["role"], row["field"]): row["actual"] for row in records if target == "all" or row["role"] == target}
    left = values(baseline); result = {}
    for variant in sorted({row["variant"] for row in rows}):
        right = values(next(row["outputs"] for row in rows if row["variant"] == variant))
        changes = [{"document_id": key[0], "role": key[1], "field": key[2], "baseline": left.get(key), "variant": right.get(key)} for key in sorted(set(left) | set(right)) if left.get(key) != right.get(key)]
        result[variant] = {"changed_count": len(changes), "changes": changes}
    (output / ("recognition_output_differences.json" if target == "all" else "mrz_output_differences.json")).write_text(json.dumps(result, indent=2), encoding="utf-8")


def full_pipeline(args: argparse.Namespace, documents: list[Any], output: Path, finalists: list[dict[str, Any]]) -> list[dict[str, Any]]:
    from benchmarks.maintained.model_matrix_benchmark import _stage_totals
    rows = []
    port = args.port
    for index, candidate in enumerate(finalists):
        config = {**BASE_ENV, "MODEL_DIR": str(args.model_dir), "VOIGHT_BENCHMARK_VISIBLE_PREPROCESSING": "", "VOIGHT_BENCHMARK_MRZ_PREPROCESSING": "", "VOIGHT_BENCHMARK_MRZ_CROP_PREPROCESSING": "", "VOIGHT_BENCHMARK_DETECTOR_PREPROCESSING": ""}
        config.update(candidate.get("env", {}))
        directory = output / "full_pipeline" / f"{index:02d}_{candidate['name']}"; directory.mkdir(parents=True, exist_ok=True)
        namespace = argparse.Namespace(port=port, model_dir=args.model_dir, timeout=args.timeout)
        server = Server(namespace, directory, config)
        lifecycle = None
        try:
            ready = server.start(); (directory / "ready.json").write_text(json.dumps(ready, indent=2), encoding="utf-8")
            for kind in DOC_TYPES:
                selected = [document for document in documents if document.document_type == kind]
                warmup, _ = _post(kind, selected, port, args.timeout); (directory / f"warmup_{kind}.json").write_text(json.dumps(warmup, indent=2), encoding="utf-8")
                baseline_snapshot = None
                for repeat in range(1, 4):
                    payload, client = _post(kind, selected, port, args.timeout)
                    score = _score(kind, selected, payload); snapshot = _snapshot(selected, payload)
                    if baseline_snapshot is None: baseline_snapshot = snapshot
                    row = {"candidate": candidate["name"], "document_type": kind, "repeat": repeat, "client_e2e_seconds": client, "server_e2e_seconds": payload.get("total_seconds"), "docs_per_second": len(selected) / client, "rss_mb": payload.get("diagnostics", {}).get("process_peak_rss_mb"), "stage_timings": _stage_totals([payload]), "output_differences": _differences(selected, baseline_snapshot, snapshot), "correctness": score}
                    rows.append(row); (directory / f"{kind}_{repeat}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        finally:
            lifecycle = server.stop(); (directory / "lifecycle.json").write_text(json.dumps(lifecycle, indent=2), encoding="utf-8")
        print(f"full {candidate['name']}: cleanup={lifecycle.get('cleanup_verified') if lifecycle else False}", flush=True)
        port += 1
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models/benchmark")
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs/benchmarks/preprocessing")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--skip-full", action="store_true")
    parser.add_argument("--reuse-output", type=Path, help="reuse a prior fixed-crop directory and continue Phases B-D")
    parser.add_argument("--full-only", type=Path, help="run Phase D from an existing output directory")
    parser.add_argument("--report-only", type=Path, help="write per-document compact tables from an existing output directory")
    parser.add_argument("--detector-only", type=Path, help="rerun Phase B from an existing fixed-crop directory")
    args = parser.parse_args()
    if args.repeats < 3: parser.error("use at least three measured repeats")
    if args.full_only:
        documents, _ = validate_and_manifest(args.dataset_root)
        finalists = json.loads((args.full_only / "finalists.json").read_text(encoding="utf-8"))["candidates"]
        rows = full_pipeline(args, documents, args.full_only, finalists)
        (args.full_only / "full_pipeline_raw.jsonl").write_text("\n".join(json.dumps(row, default=str) for row in rows) + "\n", encoding="utf-8")
        print(f"completed full-only: {args.full_only}")
        return 0
    if args.report_only:
        def read_jsonl(name: str) -> list[dict[str, Any]]:
            return [json.loads(line) for line in (args.report_only / name).read_text(encoding="utf-8").splitlines() if line.strip()]
        write_by_document_tables(args.report_only, read_jsonl("recognition_raw.jsonl"), read_jsonl("detector_raw.jsonl"), read_jsonl("mrz_raw.jsonl"))
        write_fixed_output_differences(args.report_only, "recognition_raw.jsonl", "all")
        write_fixed_output_differences(args.report_only, "mrz_raw.jsonl", "mrz")
        if (args.report_only / "full_pipeline_raw.jsonl").is_file():
            documents, _ = validate_and_manifest(args.dataset_root)
            write_full_pipeline_report(args.report_only, documents)
        print(f"completed report-only: {args.report_only}")
        return 0
    for key, value in BASE_ENV.items(): os.environ[key] = value
    os.environ["MODEL_DIR"] = str(args.model_dir)
    for key in ("VOIGHT_BENCHMARK_VISIBLE_PREPROCESSING", "VOIGHT_BENCHMARK_MRZ_PREPROCESSING", "VOIGHT_BENCHMARK_MRZ_CROP_PREPROCESSING", "VOIGHT_BENCHMARK_DETECTOR_PREPROCESSING"):
        os.environ.pop(key, None)
    documents, manifest = validate_and_manifest(args.dataset_root)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output_root / stamp; output.mkdir(parents=True, exist_ok=True)
    (output / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    settings = Settings.from_env(); models = Models(settings)
    if args.detector_only:
        documents, _ = validate_and_manifest(args.dataset_root)
        crops, detector_inputs, _ = load_fixed_crops(settings, documents, args.detector_only)
        detector_rows = run_detector_phase(documents, detector_inputs, models.text_detector(), models.text_recognizer(), crops, args.repeats, args.detector_only)
        (args.detector_only / "detector_raw.jsonl").write_text("\n".join(json.dumps(row, default=str) for row in detector_rows) + "\n", encoding="utf-8")
        detector_summary = [_median_rows(detector_rows, "B", variant) for variant in ("original", "grayscale", "contrast_1.25", "clahe_mild", "gamma_0.8", "gamma_1.2")]
        write_csv(args.detector_only / "detector_preprocessing.csv", detector_summary)
        def read_jsonl(name: str) -> list[dict[str, Any]]:
            return [json.loads(line) for line in (args.detector_only / name).read_text(encoding="utf-8").splitlines() if line.strip()]
        write_by_document_tables(args.detector_only, read_jsonl("recognition_raw.jsonl"), detector_rows, read_jsonl("mrz_raw.jsonl"))
        print(f"completed detector-only: {args.detector_only}")
        return 0
    if args.reuse_output:
        crops, detector_inputs, fixed_manifest = load_fixed_crops(settings, documents, args.reuse_output)
        (output / "fixed_crops").mkdir(parents=True, exist_ok=True)
        (output / "fixed_crops" / "manifest.json").write_text(json.dumps(fixed_manifest, indent=2), encoding="utf-8")
        for crop in crops:
            np.save(output / "fixed_crops" / f"{crop.key.replace(':', '__')}.npy", crop.image)
        for item in detector_inputs:
            np.save(output / "fixed_crops" / f"input__{item.key.replace(':', '__')}.npy", item.image)
    else:
        crops, detector_inputs, fixed_manifest = build_fixed_crops(settings, documents, models, output)
    recognizer = models.text_recognizer(); detector = models.text_detector()
    recognition_rows = []; recognition_summary = []
    for variant in VISIBLE_VARIANTS:
        rows, _ = recognize(crops, recognizer, variant, 2, args.repeats)
        recognition_rows.extend(rows); recognition_summary.append(_median_rows(rows, "A", variant))
    (output / "recognition_raw.jsonl").write_text("\n".join(json.dumps(row, default=str) for row in recognition_rows) + "\n", encoding="utf-8")
    write_csv(output / "recognition_preprocessing.csv", recognition_summary)
    detector_rows = run_detector_phase(documents, detector_inputs, detector, recognizer, crops, args.repeats, output)
    (output / "detector_raw.jsonl").write_text("\n".join(json.dumps(row, default=str) for row in detector_rows) + "\n", encoding="utf-8")
    detector_summary = [_median_rows(detector_rows, "B", variant) for variant in ("original", "grayscale", "contrast_1.25", "clahe_mild", "gamma_0.8", "gamma_1.2")]
    write_csv(output / "detector_preprocessing.csv", detector_summary)
    mrz_rows = []
    mrz_crops = [crop for crop in crops if crop.role == "mrz"]
    for variant in MRZ_VARIANTS:
        rows, _ = recognize(mrz_crops, recognizer, variant, 2, args.repeats)
        mrz_rows.extend(rows)
    (output / "mrz_raw.jsonl").write_text("\n".join(json.dumps(row, default=str) for row in mrz_rows) + "\n", encoding="utf-8")
    mrz_summary = [_median_rows(mrz_rows, "C", variant) for variant in MRZ_VARIANTS]
    write_csv(output / "mrz_preprocessing.csv", mrz_summary)
    write_by_document_tables(output, recognition_rows, detector_rows, mrz_rows)
    finalists = [{"name": "baseline", "env": {}}]
    def visible_non_passport_score(variant: str) -> tuple[float, float]:
        records = next(row["outputs"] for row in recognition_rows if row["variant"] == variant and row["repeat"] == 1)
        records = [row for row in records if row["role"] == "visible" and row["document_type"] in {"id_card", "driving_license"} and row["expected"] is not None]
        total = sum(max(1, len(_norm(row["expected"]))) for row in records)
        correct = sum(max(0, max(1, len(_norm(row["expected"]))) - _distance(_norm(row["expected"]), _norm(row["actual"]))) for row in records)
        exact = sum(_norm(row["expected"]) == _norm(row["actual"]) for row in records)
        return correct / total if total else 0.0, exact / len(records) if records else 0.0
    visible_rank = sorted((row for row in recognition_summary if row["variant"] != "original"), key=lambda row: (-visible_non_passport_score(row["variant"])[0], -visible_non_passport_score(row["variant"])[1], row["recognition_seconds"] or 1e9))[:2]
    finalists += [{"name": f"visible_{row['variant']}", "env": {"VOIGHT_BENCHMARK_VISIBLE_PREPROCESSING": row["variant"]}} for row in visible_rank]
    detector_rank = sorted((row for row in detector_summary if row["variant"] != "original" and (row["missing_lines"] or 0) <= (detector_summary[0]["missing_lines"] or 0) and (row["extra_lines"] or 0) <= (detector_summary[0]["extra_lines"] or 0)), key=lambda row: (-(row["field_character_accuracy"] or 0), row["e2e_seconds"] or 1e9))
    if detector_rank and (detector_rank[0]["field_character_accuracy"] or 0) > (detector_summary[0]["field_character_accuracy"] or 0):
        finalists.append({"name": f"detector_{detector_rank[0]['variant']}", "env": {"VOIGHT_BENCHMARK_DETECTOR_PREPROCESSING": detector_rank[0]["variant"]}})
    mrz_rank = sorted((row for row in mrz_summary if row["variant"] != "original"), key=lambda row: (-(row["mrz_exact_rate"] or 0), -(row["mrz_character_accuracy"] or 0), row["recognition_seconds"] or 1e9))[:2]
    finalists += [{"name": f"mrz_{row['variant']}", "env": {"VOIGHT_BENCHMARK_MRZ_CROP_PREPROCESSING": row["variant"]}} for row in mrz_rank]
    (output / "finalists.json").write_text(json.dumps({"selection_rule": "two visible by character accuracy, detector only on strict no-line-regression improvement, two MRZ by exact then character accuracy; this is candidate selection, not a production choice", "candidates": finalists}, indent=2), encoding="utf-8")
    full_rows = [] if args.skip_full else full_pipeline(args, documents, output, finalists)
    (output / "full_pipeline_raw.jsonl").write_text("\n".join(json.dumps(row, default=str) for row in full_rows) + ("\n" if full_rows else ""), encoding="utf-8")
    (output / "README.md").write_text("# Preprocessing benchmark\n\nPhases A-C use one fixed crop corpus produced once with CPU, 4 threads, localization batch 4, detector batch 1 at side 960, fixed-width recognition packing and batch 2. Phase D uses a fresh server, one warm-up and three measured repeats per candidate, with lifecycle evidence. `original` is the current crop; MRZ crops are the current normalized MRZ crop. Production selection is intentionally left to the operator.\n", encoding="utf-8")
    models.close()
    print(f"completed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
