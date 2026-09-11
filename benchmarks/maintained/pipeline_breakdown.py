"""CPU measurement suite for the current Voight document pipelines.

The primary suite is deliberately benchmark-only.  P0/I0/D0 call the real
``/v1`` routes; ablations call the same localizers, preparation, Paddle
adapters, profile parsers, and MRZ parser directly.  No model is downloaded by
this module and no production setting is changed.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import cv2
import numpy as np
import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.api.v1 import _run_batch
from app.artifacts import ArtifactWriter
from app.config import Settings
from app.documents.driving_license_fields import parse_fields, validation_warnings
from app.documents.identity import _parse_visible, _required_warnings
from app.documents.mrz import MrzProfile, crop_polygon, preprocess, reconstruct, select, parse as parse_mrz
from app.documents.passport_localization import page_corners_from_mrz_width
from app.documents.profiles import load_document_profile
from app.inference.batch import BatchedOcr, OcrSample, ProfileBatchItem, ProfileBatchRunner, _line_crop, _pad_detection_batch
from app.inference.packing import recognition_batch_packer
from app.models import Models
from app.pipeline import RegionProfile, complete_profile, load_region_profile, prepare_profile_from_detection
from app.uploads import document_from_bytes

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
DOC_TYPES = ("passport", "id_card", "driving_license")
PRIMARY_CONFIG = {
    "RUNTIME_TARGET": "cpu", "CPU_THREADS": 4, "TEXT_RECOGNITION_PROCESSES": 1,
    "LOCALIZATION_BATCH_SIZE": 16, "TEXT_DETECTION_BATCH_SIZE": 16,
    "TEXT_RECOGNITION_BATCH_SIZE": 32, "MRZ_RECOGNITION_BATCH_SIZE": 16,
    "TEXT_RECOGNITION_PACKING": "fixed-width", "TEXT_DETECTOR_MODEL": "PP-OCRv6_medium_det",
    "TEXT_RECOGNIZER_MODEL": "latin_PP-OCRv5_mobile_rec", "MRZ_RECOGNIZER_BACKEND": "generic-paddle",
    "DOCALIGNER_MODEL": "fastvit_sa24",
}
VARIANTS = {
    "passport": ("passport_full_production", "passport_visible_no_mrz_ocr", "passport_mrz_only", "passport_localization_preparation_only"),
    "id_card": ("id_card_full_production", "id_card_visible_probe", "id_card_visible_known_side", "id_card_mrz_probe", "id_card_mrz_known_back", "id_card_side_visible_breakdown"),
    "driving_license": ("driving_license_full_production", "driving_license_localization_preparation_only", "driving_license_visible_ocr_only"),
}
ALL_VARIANTS = tuple(variant for variants in VARIANTS.values() for variant in variants)
ARTIFACTS = ArtifactWriter(ROOT / "outputs", "", "benchmark", False)


@dataclass(frozen=True)
class Document:
    document_type: str
    document_id: str
    paths: tuple[tuple[str, Path], ...]
    annotation: Path

    @property
    def physical_count(self) -> int:
        return len(self.paths)


@dataclass
class Run:
    variant: str
    document_type: str
    repeat: int
    logical_count: int
    physical_count: int
    source_ids: tuple[str, ...]
    status: str
    total_seconds: float
    client_seconds: float | None
    server_seconds: float | None
    stages: dict[str, float]
    diagnostics: dict[str, Any]
    outputs: dict[str, dict[str, Any]]
    error: str | None = None


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_dataset(root: Path) -> list[Document]:
    documents: list[Document] = []
    for kind in ("passport", "driving_license"):
        for image in sorted((root / kind).glob("*")):
            if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                documents.append(Document(kind, image.stem, (("image", image),), root / "annotations" / kind / f"{image.stem}.json"))
    for directory in sorted((root / "id_card").glob("*")):
        if not directory.is_dir():
            continue
        sides = []
        for side in ("front", "back"):
            matches = sorted(p for p in directory.glob(f"{side}.*") if p.suffix.lower() in IMAGE_EXTENSIONS)
            if len(matches) != 1:
                raise ValueError(f"{directory}: expected one {side} image")
            sides.append((side, matches[0]))
        documents.append(Document("id_card", directory.name, tuple(sides), root / "annotations" / "id_card" / f"{directory.name}.json"))
    return sorted(documents, key=lambda d: (DOC_TYPES.index(d.document_type), d.document_id.lower()))


def validate_and_manifest(root: Path) -> tuple[list[Document], dict[str, Any]]:
    from scripts.dataset.annotate import validate_dataset

    issues = validate_dataset(root)
    if issues:
        raise ValueError("dataset validation failed: " + "; ".join(issues))
    documents = discover_dataset(root)
    entries = []
    for document in documents:
        annotation_bytes = document.annotation.read_bytes()
        images = []
        annotation = json.loads(annotation_bytes)
        for role, path in document.paths:
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"cannot decode {path}")
            images.append({"role": role, "sha256": sha256_file(path), "width": int(image.shape[1]), "height": int(image.shape[0]), "channels": int(image.shape[2]) if image.ndim == 3 else 1})
        entries.append({
            "document_id": document.document_id, "document_type": document.document_type,
            "physical_image_count": document.physical_count, "images": images,
            "annotation_sha256": sha256_bytes(annotation_bytes),
            "annotation_image_sha256": annotation.get("image_sha256", {}),
        })
    return documents, {"schema_version": 1, "source_root": str(root), "documents": entries, "counts": {kind: sum(d.document_type == kind for d in documents) for kind in DOC_TYPES}, "physical_images": sum(d.physical_count for d in documents)}


def cycles(documents: list[Document], kind: str, n: int) -> list[Document]:
    source = [d for d in documents if d.document_type == kind]
    if not source:
        return []
    return [source[index % len(source)] for index in range(n)]


def consumed_physical_count(kind: str, variant: str, documents: list[Document], diagnostics: dict[str, Any] | None = None) -> int:
    if kind != "id_card" or variant not in {"id_card_mrz_probe", "id_card_mrz_known_back"}:
        return sum(document.physical_count for document in documents)
    if variant == "id_card_mrz_known_back":
        return len(documents)
    fallback = (diagnostics or {}).get("localization", {}).get("mrz_fallback", {})
    return len(documents) + sum(fallback.get("submitted_batch_sizes", []))


def stats(values: Iterable[float]) -> dict[str, float | None]:
    values = list(values)
    if not values:
        return {"median": None, "min": None, "max": None, "mad": None, "iqr": None}
    ordered = sorted(values)
    median = statistics.median(ordered)
    deviations = [abs(value - median) for value in ordered]
    quartiles = statistics.quantiles(ordered, n=4, method="inclusive") if len(ordered) > 1 else [median] * 3
    return {"median": median, "min": min(ordered), "max": max(ordered), "mad": statistics.median(deviations), "iqr": quartiles[2] - quartiles[0]}


def _stage_values(diagnostics: dict[str, Any], total: float) -> dict[str, float]:
    localization = sum(float(v.get("wall_seconds", 0.0)) for v in diagnostics.get("localization", {}).values() if isinstance(v, dict))
    pipeline = diagnostics.get("pipeline", {})
    detection = float(diagnostics.get("text_detection", diagnostics.get("visible_ocr", {}).get("text_detection", diagnostics.get("mrz_ocr", {}).get("text_detection", {}))).get("wall_seconds", 0.0))
    recognition = float(diagnostics.get("text_recognition", diagnostics.get("visible_ocr", {}).get("text_recognition", diagnostics.get("mrz_ocr", {}).get("text_recognition", {}))).get("wall_seconds", 0.0))
    mrz_work = float(pipeline.get("mrz_crop_preprocess_seconds", 0.0)) + float(diagnostics.get("mrz_recognition", {}).get("wall_seconds", 0.0))
    values = {
        "localization": localization,
        "canonicalization": float(pipeline.get("canonicalization_seconds", 0.0)),
        "detection": detection,
        "roi_filtering_cropping": float(diagnostics.get("line_crop_seconds", 0.0)),
        "recognition": recognition,
        "mrz_work": mrz_work,
        "parsing_validation": float(pipeline.get("parsing_validation_seconds", 0.0)),
    }
    values["other"] = max(0.0, total - sum(values.values()))
    return values


def _batch_details(diagnostics: dict[str, Any]) -> dict[str, Any]:
    """Return shape/batch evidence without copying OCR text or image data."""
    stages = diagnostics.get("diagnostics", diagnostics)

    def calls(name: str) -> list[dict[str, Any]]:
        stage = stages.get(name, {})
        if not isinstance(stage, dict):
            return []
        return [
            {
                key: call[key]
                for key in (
                    "role", "submitted_batch_size", "tensor_batch_size", "tensor_batch_sizes",
                    "input_widths", "input_heights", "submitted_input_shapes",
                    "padded_tensor_pixel_area", "padding_efficiency", "shape_metric_source",
                    "tensor_shapes", "tensor_pixel_counts", "detector_resized_shapes",
                    "detector_resize_config", "polygon_coordinate_space",
                )
                if key in call
            }
            for call in stage.get("calls", ())
        ]

    line_filter = stages.get("line_filter", {})
    return {
        "detection": calls("text_detection"),
        "recognition": calls("text_recognition"),
        "mrz_recognition": calls("mrz_recognition"),
        "recognition_packing": stages.get("text_recognition", {}).get("packing_strategy"),
        "recognition_crop_count": line_filter.get("recognition_candidate_count", 0),
        "filtered_line_count": line_filter.get("filtered_before_recognition_count", 0),
        "detected_line_count": line_filter.get("detected_line_count", 0),
        "process_peak_rss_mb": stages.get("process_peak_rss_mb"),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(v) for v in value]
    return value


def _settings() -> Settings:
    load_dotenv(ROOT / ".env", override=False)
    settings = Settings.from_env()
    actual = {
        "RUNTIME_TARGET": settings.runtime.target, "CPU_THREADS": settings.runtime.cpu_threads,
        "TEXT_RECOGNITION_PROCESSES": settings.runtime.text_recognition_processes,
        "LOCALIZATION_BATCH_SIZE": settings.runtime.localization_batch_size,
        "TEXT_DETECTION_BATCH_SIZE": settings.runtime.text_detection_batch_size,
        "TEXT_RECOGNITION_BATCH_SIZE": settings.runtime.text_recognition_batch_size,
        "MRZ_RECOGNITION_BATCH_SIZE": settings.runtime.mrz_recognition_batch_size,
        "TEXT_RECOGNITION_PACKING": settings.runtime.text_recognition_packing,
        "TEXT_DETECTOR_MODEL": settings.models.text_detector.model,
        "TEXT_RECOGNIZER_MODEL": settings.models.text_recognizer.model,
        "MRZ_RECOGNIZER_BACKEND": settings.models.mrz.recognizer_backend,
        "DOCALIGNER_MODEL": settings.driving_license.aligner_model,
    }
    if actual != PRIMARY_CONFIG and os.getenv("BENCHMARK_ALLOW_CONFIG_OVERRIDE") != "1":
        raise ValueError(f"benchmark profile mismatch: expected {PRIMARY_CONFIG}, got {actual}")
    if settings.runtime.target != "cpu" or settings.ocr.device != "cpu":
        raise ValueError("benchmark suite is CPU-only")
    return settings


def environment(settings: Settings, manifest: dict[str, Any], started: str) -> dict[str, Any]:
    def text_output(value: str | bytes) -> str:
        return value.decode("utf-8", errors="replace") if isinstance(value, bytes) else value

    def distribution_version(*names: str) -> str | None:
        for name in names:
            try:
                return importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError:
                continue
        return None

    try:
        affinity = sorted(os.sched_getaffinity(0))
    except AttributeError:
        affinity = None
    cpu = {}
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu["model"] = line.split(":", 1)[1].strip(); break
    except OSError:
        pass
    versions = {
        "paddle": distribution_version("paddlepaddle", "paddlepaddle-gpu"),
        "paddleocr": distribution_version("paddleocr"),
        "onnxruntime": distribution_version("onnxruntime", "onnxruntime-gpu"),
        "cv2": distribution_version("opencv-python-headless", "opencv-python"),
    }
    physical = None
    try:
        lines = text_output(subprocess.check_output(["lscpu", "-p=CPU,CORE,SOCKET"], text=True)).splitlines()
        physical = len({tuple(line.split(",")[1:3]) for line in lines if line and not line.startswith("#")})
    except (OSError, subprocess.SubprocessError):
        pass
    return {"git_commit": text_output(subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)).strip(), "git_dirty": bool(text_output(subprocess.check_output(["git", "status", "--porcelain"], text=True)).strip()), "os": platform.platform(), "kernel": platform.release(), "python": sys.version, "packages": versions, "cpu": {**cpu, "physical_cores": physical, "logical_cpus": os.cpu_count(), "affinity": affinity}, "runtime": {"target": settings.runtime.target, "ocr_device": settings.ocr.device, "cpu_threads": settings.runtime.cpu_threads, "localization_batch_size": settings.runtime.localization_batch_size, "text_detection_batch_size": settings.runtime.text_detection_batch_size, "text_recognition_batch_size": settings.runtime.text_recognition_batch_size, "mrz_recognition_batch_size": settings.runtime.mrz_recognition_batch_size, "recognition_packing": settings.runtime.text_recognition_packing}, "models": {"detector": settings.models.text_detector.model, "recognizer": settings.models.text_recognizer.model, "docaligner": settings.driving_license.aligner_model, "mrz_backend": settings.models.mrz.recognizer_backend, "mrz_model": settings.models.mrz.recognizer_model}, "preprocessing": {"ocr_predict_config": {"text_det_thresh": 0.30, "text_det_box_thresh": 0.50, "text_det_unclip_ratio": 2.0}, "ocr_max_side": settings.mrz.max_side, "ocr_contrast": settings.mrz.contrast, "mrz_polygon_padding_ratio": settings.mrz.polygon_padding_ratio}, "artifacts_enabled": settings.artifacts.enabled, "benchmark_started_utc": started, "manifest_counts": manifest["counts"]}


def _read_images(document: Document) -> dict[str, np.ndarray]:
    result = {}
    for role, path in document.paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"cannot decode {path}")
        result[role] = image
    return result


def _profiles(settings: Settings, kind: str) -> dict[str, Any]:
    if kind == "driving_license":
        return {"image": load_region_profile(settings.driving_license.data_crop, settings.driving_license.field_rois)}
    profile = load_document_profile(settings.profiles.passport if kind == "passport" else settings.profiles.id_card)
    return {region: RegionProfile(value["data_crop"], value["field_rois"]) for region, value in profile["regions"].items()}


def variant_scope(kind: str, variant: str) -> dict[str, bool]:
    """Declare timed work for tests and for the benchmark report."""
    return {
        "visible_ocr": variant in {"passport_visible_no_mrz_ocr", "id_card_visible_probe", "id_card_visible_known_side", "id_card_side_visible_breakdown", "driving_license_visible_ocr_only"},
        "mrz_ocr": variant in {"passport_mrz_only", "id_card_mrz_probe", "id_card_mrz_known_back"},
        "preparation": variant not in {"id_card_mrz_probe", "id_card_mrz_known_back", "passport_mrz_only"},
    }


def _items(settings: Settings, documents: list[Document], kind: str, variant: str) -> tuple[list[ProfileBatchItem], dict[str, Any], dict[str, np.ndarray]]:
    profile_path = settings.profiles.passport if kind == "passport" else settings.profiles.id_card
    doc_profile = None if kind == "driving_license" else load_document_profile(profile_path)
    loaded: dict[str, np.ndarray] = {}
    jobs: list[ProfileBatchItem] = []
    owners: dict[str, Any] = {}
    side_known = variant in {"id_card_visible_known_side", "id_card_mrz_known_back"}
    for index, document in enumerate(documents):
        images = _read_images(document)
        for role, image in images.items():
            item_id = f"{index}:{document.document_id}:{role}"
            loaded[item_id] = image
            if kind == "driving_license":
                profile = load_region_profile(settings.driving_license.data_crop, settings.driving_license.field_rois)
                width, height, parser, validator, local_kind = settings.driving_license.canonical_width, settings.driving_license.canonical_height, parse_fields, validation_warnings, "docaligner"
                mrz_profile = None; probe = False; fallback = None; page = None
            else:
                region = "data_page" if kind == "passport" else role
                profile = RegionProfile(doc_profile["regions"][region]["data_crop"], doc_profile["regions"][region]["field_rois"])
                width, height = int(doc_profile["canonical_size"]["width"]), int(doc_profile["canonical_size"]["height"])
                parser = _parse_visible; validator = _required_warnings(doc_profile, region)
                is_passport = kind == "passport"
                local_kind = "mrz" if is_passport else "docaligner"
                mrz_profile = MrzProfile((2,)) if is_passport else (MrzProfile((3,)) if role == "back" else None)
                if variant in {"passport_visible_no_mrz_ocr", "passport_mrz_only", "passport_localization_preparation_only"}:
                    pass
                probe = kind == "id_card" and role == "back" and not side_known
                fallback = f"{index}:{document.document_id}:back" if kind == "id_card" and role == "front" and not side_known else None
                page = doc_profile.get("document_localization", {}).get("page_corners_relative_to_mrz_width") if is_passport else None
            jobs.append(ProfileBatchItem(item_id, image, profile, local_kind, parser, validator, ARTIFACTS, width, height, settings.driving_license.aligner_padding, settings.driving_license.min_overlap_ratio, page, mrz_profile, probe, fallback))
            owners[item_id] = (document.document_id, role)
    return jobs, owners, loaded


def _localize(localizer: Any, jobs: list[tuple[str, np.ndarray]], batch_size: int, *, pad: int = 0) -> tuple[dict[str, Any], dict[str, Any]]:
    results: dict[str, Any] = {}
    stage = {"configured_batch_size": batch_size, "calls": [], "tensor_batch_sizes": []}
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start:start + batch_size]
        inputs = []
        for _, image in chunk:
            inputs.append(cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_CONSTANT) if pad else image)
        began = time.perf_counter(); values = list(localizer.localize_batch(inputs)); elapsed = time.perf_counter() - began
        if len(values) != len(chunk):
            raise ValueError("localizer returned the wrong number of values")
        tensor_size = int(getattr(localizer, "last_tensor_batch_size", len(chunk)))
        stage["calls"].append({"submitted_batch_size": len(chunk), "tensor_batch_size": tensor_size, "model_seconds": float(getattr(localizer, "last_model_seconds", elapsed)), "wall_seconds": elapsed, "failure_count": 0})
        stage["tensor_batch_sizes"].append(tensor_size)
        results.update({item_id: value for (item_id, _), value in zip(chunk, values)})
    stage["model_call_count"] = len(stage["calls"]); stage["submitted_batch_sizes"] = [c["submitted_batch_size"] for c in stage["calls"]]; stage["wall_seconds"] = sum(c["wall_seconds"] for c in stage["calls"]); stage["model_seconds"] = sum(c["model_seconds"] for c in stage["calls"])
    return results, stage


def _ocr(models: Models, samples: list[OcrSample], settings: Settings, *, detection_size: int | None = None, recognition_size: int | None = None) -> Any:
    runner = BatchedOcr(models.text_detector(), models.text_recognizer(), detection_batch_size=detection_size or settings.runtime.text_detection_batch_size, recognition_batch_size=recognition_size or settings.runtime.text_recognition_batch_size, recognition_packer=recognition_batch_packer(settings.runtime.text_recognition_packing))
    return runner.run(samples)


def run_partial(settings: Settings, models: Models, documents: list[Document], kind: str, variant: str) -> tuple[float, dict[str, Any], dict[str, dict[str, Any]]]:
    started = time.perf_counter()
    jobs, owners, loaded = _items(settings, documents, kind, variant)
    scope = variant_scope(kind, variant)
    use_visible = scope["visible_ocr"]
    use_mrz = scope["mrz_ocr"]
    localizers = {}
    if kind == "driving_license" or use_visible:
        localizers["docaligner"] = models.document_localizer()
    if kind != "driving_license" and (kind == "passport" or use_mrz or variant == "id_card_visible_probe"):
        localizers["mrz"] = models.mrz_localizer()
    if variant.endswith("side_visible_breakdown"):
        use_visible = True
    localizations: dict[str, Any] = {}; localization_diag: dict[str, Any] = {}
    docaligner_jobs = [(job.item_id, job.image) for job in jobs if job.localization_kind == "docaligner"]
    mrz_jobs = [(job.item_id, job.image) for job in jobs if job.localization_kind == "mrz"]
    if kind == "id_card" and use_visible:
        doc_locations, docaligner_diag = _localize(localizers["docaligner"], docaligner_jobs, settings.runtime.localization_batch_size, pad=settings.driving_license.aligner_padding)
        localizations.update({f"doc:{key}": value for key, value in doc_locations.items()})
        localization_diag["docaligner"] = docaligner_diag
        if variant in {"id_card_visible_known_side", "id_card_side_visible_breakdown"}:
            pass
        else:
            back = [(job.item_id, job.image) for job in jobs if job.mrz_profile is not None and job.item_id.endswith(":back")]
            probe, probe_diag = _localize(localizers["mrz"], back, settings.runtime.localization_batch_size)
            localizations.update({f"mrz:{key}": value for key, value in probe.items()}); localization_diag["mrz"] = probe_diag
            fallback = [(job.item_id.replace(":back", ":front"), loaded[job.item_id.replace(":back", ":front")]) for job in jobs if job.item_id.endswith(":back") and localizations[f"mrz:{job.item_id}"].polygon.size != 8]
            if fallback:
                fallback_result, fallback_diag = _localize(localizers["mrz"], fallback, settings.runtime.localization_batch_size)
                localizations.update({f"mrz:{key}": value for key, value in fallback_result.items()}); localization_diag["mrz_fallback"] = fallback_diag
    elif kind == "id_card" and use_mrz:
        back = [(job.item_id, job.image) for job in jobs if job.item_id.endswith(":back")]
        probe, probe_diag = _localize(localizers["mrz"], back, settings.runtime.localization_batch_size); localizations.update({f"mrz:{key}": value for key, value in probe.items()}); localization_diag["mrz"] = probe_diag
        fallback = [(job.item_id.replace(":back", ":front"), loaded[job.item_id.replace(":back", ":front")]) for job in jobs if job.item_id.endswith(":back") and localizations[f"mrz:{job.item_id}"].polygon.size != 8]
        if fallback:
            fallback_result, fallback_diag = _localize(localizers["mrz"], fallback, settings.runtime.localization_batch_size); localizations.update({f"mrz:{key}": value for key, value in fallback_result.items()}); localization_diag["mrz_fallback"] = fallback_diag
    elif kind == "passport":
        targets = [(job.item_id, job.image) for job in jobs]
        mrz_locations, localization_diag["mrz"] = _localize(localizers["mrz"], targets, settings.runtime.localization_batch_size)
        localizations.update({f"mrz:{key}": value for key, value in mrz_locations.items()})
    elif kind == "driving_license":
        targets = [(job.item_id, job.image) for job in jobs]
        doc_locations, localization_diag["docaligner"] = _localize(localizers["docaligner"], targets, settings.runtime.localization_batch_size, pad=settings.driving_license.aligner_padding)
        localizations.update({f"doc:{key}": value for key, value in doc_locations.items()})

    prepared: dict[str, Any] = {}; mrz_polygons: dict[str, np.ndarray] = {}; prep_seconds = 0.0
    for job in jobs:
        if f"mrz:{job.item_id}" in localizations and localizations[f"mrz:{job.item_id}"].polygon.size == 8:
            mrz_polygons[job.item_id] = localizations[f"mrz:{job.item_id}"].polygon.reshape(4, 2)
        location_key = f"doc:{job.item_id}" if f"doc:{job.item_id}" in localizations else f"mrz:{job.item_id}"
        if location_key not in localizations or not scope["preparation"]:
            continue
        try:
            location = localizations[location_key].polygon.reshape(4, 2)
            if job.localization_kind == "mrz":
                corners = page_corners_from_mrz_width(location, job.passport_page_corners)
                padded = None; padded_corners = corners
            else:
                corners = location - job.padding; padded = cv2.copyMakeBorder(job.image, job.padding, job.padding, job.padding, job.padding, cv2.BORDER_CONSTANT) if job.padding else None; padded_corners = corners + job.padding
            prep_started = time.perf_counter()
            prepared[job.item_id] = prepare_profile_from_detection(job.image, job.profile, corners, ARTIFACTS, canonical_width=job.canonical_width, canonical_height=job.canonical_height, padding=job.padding, padded=padded, padded_corners=padded_corners, started_total=started)
            prep_seconds += time.perf_counter() - prep_started
        except (cv2.error, KeyError, TypeError, ValueError):
            continue

    timed_started = time.perf_counter() if variant == "driving_license_visible_ocr_only" else started

    visible_result = None; mrz_result = None; mrz_crop_seconds = 0.0
    if use_visible:
        samples = [OcrSample(f"visible:{job.item_id}", prepared[job.item_id].data_crop, prepared[job.item_id].profile.field_rois) for job in jobs if job.item_id in prepared]
        visible_result = _ocr(models, samples, settings)
    if use_mrz:
        mrz_samples = []
        for job in jobs:
            if job.mrz_profile is None or job.item_id not in mrz_polygons:
                continue
            crop_started = time.perf_counter(); crop, _ = crop_polygon(job.image, mrz_polygons[job.item_id], settings.mrz.polygon_padding_ratio); processed = preprocess(crop, settings.mrz.max_side, settings.mrz.contrast); mrz_crop_seconds += time.perf_counter() - crop_started
            mrz_samples.append(OcrSample(f"mrz:{job.item_id}", processed))
        mrz_result = _ocr(models, mrz_samples, settings)
    assembly_started = time.perf_counter()
    parse_validation_seconds = 0.0
    outputs: dict[str, dict[str, Any]] = {document.document_id: {"fields": {}, "mrz": []} for document in documents}
    for job in jobs:
        document_id, role = owners[job.item_id]
        if visible_result and job.item_id in prepared:
            tokens = visible_result.tokens.get(f"visible:{job.item_id}", [])
            values, report = complete_profile(prepared[job.item_id], tokens, job.parse_fields, job.validate_fields, min_overlap=job.min_overlap, ocr_seconds=None)
            parse_validation_seconds += sum(float(report["timings"].get(name, 0.0)) for name in ("field_assignment_seconds", "field_parsing_seconds", "validation_seconds"))
            outputs[document_id]["fields"].update(values)
        if mrz_result and job.mrz_profile is not None:
            tokens = mrz_result.tokens.get(f"mrz:{job.item_id}", [])
            outputs[document_id]["mrz"] = [line.text for line in select(reconstruct(tokens), job.mrz_profile.line_counts)]
    diagnostics = {"localization": localization_diag, "pipeline": {"document_preparation_seconds": prep_seconds, "canonicalization_seconds": sum(p.timings.get("canonicalization_seconds", 0.0) for p in prepared.values()), "data_crop_seconds": sum(p.timings.get("data_crop_seconds", 0.0) for p in prepared.values()), "mrz_crop_preprocess_seconds": mrz_crop_seconds, "parsing_validation_seconds": parse_validation_seconds, "result_assembly_seconds": time.perf_counter() - assembly_started}, "variant_scope": variant}
    if variant == "driving_license_visible_ocr_only":
        diagnostics["preparation_outside_timed_region"] = True
    if visible_result:
        diagnostics.update(visible_result.diagnostics); diagnostics["visible_ocr"] = {"text_detection": visible_result.diagnostics["text_detection"], "text_recognition": visible_result.diagnostics["text_recognition"]}
    if mrz_result:
        diagnostics["mrz_ocr"] = {"text_detection": mrz_result.diagnostics["text_detection"], "text_recognition": mrz_result.diagnostics["text_recognition"]}
    return time.perf_counter() - timed_started, diagnostics, outputs


def _input_for(document: Document) -> Any:
    files = tuple(document_from_bytes(path.read_bytes(), path.name, "image/jpeg") for _, path in document.paths)
    from app.api.v1 import LogicalInput
    from app.contracts import DocumentInput, DocumentType
    dtype = DocumentType(document.document_type)
    if dtype == DocumentType.ID_CARD:
        source = DocumentInput(document_type=dtype, front="front", back="back")
    else:
        source = DocumentInput(document_type=dtype, image=document.document_id)
    return LogicalInput(source, files)


def _digest(output: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(output, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _response_outputs(response: Any) -> dict[str, dict[str, Any]]:
    outputs = {}
    for item in response.items:
        document_id = item.input.image or item.input.front or str(item.index)
        if not item.success:
            outputs[document_id] = {"fields": {}, "mrz": []}; continue
        outputs[document_id] = {"fields": {name: field.value for name, field in item.result.fields.items()}, "mrz": list(item.result.mrz.raw_lines)}
    return outputs


def run_production(settings: Settings, documents: list[Document], kind: str, repeat: int, base_url: str, timeout: float) -> Run:
    started = time.perf_counter(); source_ids = tuple(d.document_id for d in documents); files: Any
    if kind == "id_card":
        import io, zipfile
        body = io.BytesIO()
        with zipfile.ZipFile(body, "w") as archive:
            for index, document in enumerate(documents):
                for role, path in document.paths:
                    archive.writestr(f"card-{index:03d}/{role}{path.suffix.lower()}", path.read_bytes())
        files = {"archive": ("cards.zip", body.getvalue(), "application/zip")}
    else:
        path_index = 0
        files = []
        for document in documents:
            path = documents[path_index].paths[0][1]; path_index += 1
            mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
            files.append(("images", (f"{document.document_id}{path.suffix.lower()}", path.read_bytes(), mime)))
    try:
        response = requests.post(f"{base_url.rstrip('/')}/v1/ocr/{kind.replace('_', '-')}/batch", files=files, timeout=timeout)
        client_seconds = time.perf_counter() - started
        if not response.ok:
            return Run(f"{kind}_full_production", kind, repeat, len(documents), sum(d.physical_count for d in documents), source_ids, "failed", client_seconds, client_seconds, None, {}, {}, {}, response.text[:500])
        payload = response.json(); server = float(payload["total_seconds"]); diagnostics = payload.get("diagnostics", {}); outputs = {}
        # Keep only non-sensitive output signatures and correctness inputs are read from this payload in memory.
        for index, item in enumerate(payload.get("items", [])):
            identifier = documents[index].document_id if index < len(documents) else str(item.get("index"))
            result = item.get("result") or {}
            outputs[identifier] = {"fields": {name: value.get("value") for name, value in result.get("fields", {}).items()}, "mrz": (result.get("mrz") or {}).get("raw_lines", [])}
        return Run(f"{kind}_full_production", kind, repeat, len(documents), sum(d.physical_count for d in documents), source_ids, "ok", server, client_seconds, server, _stage_values(diagnostics, server), {"diagnostics": diagnostics, "payload": payload}, outputs)
    except (requests.RequestException, ValueError, KeyError) as error:
        return Run(f"{kind}_full_production", kind, repeat, len(documents), sum(d.physical_count for d in documents), source_ids, "failed", time.perf_counter() - started, time.perf_counter() - started, None, {}, {}, {}, str(error))


def annotation_truth(document: Document) -> dict[str, Any]:
    return json.loads(document.annotation.read_text(encoding="utf-8"))


def _distance(left: str, right: str) -> int:
    row = list(range(len(right) + 1))
    for i, a in enumerate(left, 1):
        next_row = [i]
        for j, b in enumerate(right, 1):
            next_row.append(min(next_row[-1] + 1, row[j] + 1, row[j - 1] + (a != b)))
        row = next_row
    return row[-1]


def score(documents: list[Document], runs: list[Run]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for run in runs:
        if run.status != "ok": continue
        by_id = {d.document_id: d for d in cycles(documents, run.document_type, run.logical_count)}
        visible = {"evaluated": 0, "exact": 0, "characters": 0, "character_total": 0, "missing": 0, "incorrect": 0}
        mrz = {"documents": 0, "full_exact": 0, "lines": 0, "line_exact": 0, "characters": 0, "character_total": 0, "parser_success": 0, "validation_success": 0}
        for document_id, output in run.outputs.items():
            document = by_id.get(document_id)
            if not document: continue
            truth = annotation_truth(document)
            for name, entry in truth.get("fields", {}).items():
                if not isinstance(entry, dict) or entry.get("state") == "unreadable" or entry.get("state") not in {"value", "empty"}: continue
                expected = entry.get("value") if entry.get("state") == "value" else None; actual = output.get("fields", {}).get(name)
                expected_text, actual_text = "" if expected is None else str(expected), "" if actual is None else str(actual)
                visible["evaluated"] += 1; visible["character_total"] += max(len(expected_text), 1); visible["characters"] += max(len(expected_text), 1) - _distance(expected_text, actual_text)
                if actual == expected or (expected is None and actual in (None, "")): visible["exact"] += 1
                elif actual in (None, ""): visible["missing"] += 1
                else: visible["incorrect"] += 1
            lines = [line for line in truth.get("mrz", {}).get("lines", []) if isinstance(line, str)]
            if lines:
                actual_lines = output.get("mrz", []); mrz["documents"] += 1; full = len(actual_lines) == len(lines)
                for index, expected in enumerate(lines):
                    actual = actual_lines[index] if index < len(actual_lines) else ""; mrz["lines"] += 1; mrz["character_total"] += len(expected); mrz["characters"] += len(expected) - _distance(expected, actual)
                    if actual == expected: mrz["line_exact"] += 1
                    else: full = False
                if full: mrz["full_exact"] += 1
                parsed = parse_mrz("\n".join(actual_lines), run.document_type)
                if parsed.raw_lines: mrz["parser_success"] += 1
                if parsed.validations and all(v.status.value == "passed" for v in parsed.validations): mrz["validation_success"] += 1
        result[run.variant + f"@{run.repeat}:{run.logical_count}"] = {"visible": visible, "mrz": mrz}
    return result


def stability(runs: list[Run]) -> dict[str, Any]:
    grouped: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for run in runs:
        if run.status != "ok":
            continue
        for source_id, output in run.outputs.items():
            grouped.setdefault((run.variant, run.document_type, run.logical_count, source_id), []).append(output)
    documents = fields = mrz_lines = 0
    for values in grouped.values():
        if len({_digest(value) for value in values}) > 1:
            documents += 1
        field_names = set().union(*(value.get("fields", {}) for value in values))
        fields += sum(len({json.dumps(value.get("fields", {}).get(name), sort_keys=True) for value in values}) > 1 for name in field_names)
        max_lines = max((len(value.get("mrz", [])) for value in values), default=0)
        mrz_lines += sum(len({value.get("mrz", [])[index] if index < len(value.get("mrz", [])) else None for value in values}) > 1 for index in range(max_lines))
    return {"documents_with_unstable_output": documents, "fields_with_unstable_output": fields, "mrz_lines_with_unstable_output": mrz_lines}


def _counts(diagnostics: dict[str, Any]) -> dict[str, Any]:
    stages = diagnostics.get("diagnostics", diagnostics)
    def stage(name: str) -> dict[str, Any]: return stages.get(name, {}) if isinstance(stages.get(name, {}), dict) else {}
    return {"configured_batch_sizes": {name: stage(name).get("configured_batch_size") for name in ("text_detection", "text_recognition")}, "tensor_batch_sizes": {name: stage(name).get("tensor_batch_sizes", []) for name in ("text_detection", "text_recognition")}, "model_call_counts": {name: stage(name).get("model_call_count", 0) for name in ("text_detection", "text_recognition")}}


def microbench(models: Models, settings: Settings, documents: list[Document], output: Path) -> list[dict[str, Any]]:
    images = [image for document in documents for image in _read_images(document).values()]
    corpus = {"localization": images, "detection_visible": [], "detection_mrz": [], "recognition_visible": [], "recognition_mrz": []}
    for image in images:
        corpus["detection_visible"].append(image)
    # Build real canonical and MRZ crops once, outside timed microbenchmarks.
    for document in documents:
        if document.document_type == "driving_license":
            continue
        for role, image in _read_images(document).items():
            if document.document_type == "id_card" and role == "front":
                continue
            try:
                localizer = models.mrz_localizer(); loc = localizer.localize_batch([image])[0]
                polygon = loc.polygon.reshape(4, 2); crop, _ = crop_polygon(image, polygon, settings.mrz.polygon_padding_ratio); corpus["detection_mrz"].append(preprocess(crop, settings.mrz.max_side, settings.mrz.contrast))
            except Exception:
                continue
    for image in corpus["detection_visible"]:
        try:
            detected = models.text_detector().detect_batch([image])[0]
            for region in detected.regions:
                line, _ = _line_crop(image, region.polygon)
                corpus["recognition_visible"].append(line)
        except Exception:
            continue
    for image in corpus["detection_mrz"]:
        try:
            detected = models.text_detector().detect_batch([image])[0]
            for region in detected.regions:
                line, _ = _line_crop(image, region.polygon)
                corpus["recognition_mrz"].append(line)
        except Exception:
            continue
    def bench(kind: str, values: list[np.ndarray], batch_size: int, mode: str) -> dict[str, Any]:
        if not values: return {"stage": kind, "mode": mode, "batch_size": batch_size, "status": "unsupported", "input_count": 0}
        model = models.text_detector() if mode == "detection" else models.text_recognizer(); calls = []; began = time.perf_counter()
        for start in range(0, len(values), batch_size):
            chunk = values[start:start + batch_size]; inputs = _pad_detection_batch(list(enumerate(chunk))) if mode == "detection" else [(i, v) for i, v in enumerate(chunk)]
            actual = [v for _, v in inputs]; call_start = time.perf_counter(); (model.detect_batch(actual) if mode == "detection" else model.recognize_batch(actual)); elapsed = time.perf_counter() - call_start; calls.append({"submitted_batch_size": len(actual), "tensor_batch_size": len(actual), "wall_seconds": elapsed})
        total = time.perf_counter() - began; return {"stage": kind, "mode": mode, "batch_size": batch_size, "status": "ok", "input_count": len(values), "line_dimensions": [{"width": int(v.shape[1]), "height": int(v.shape[0])} for v in values[:100]], "calls": calls, "model_call_count": len(calls), "tensor_batch_sizes": [c["tensor_batch_size"] for c in calls], "seconds": total, "throughput_per_second": len(values) / total, "ms_per_input": total * 1000 / len(values), "packing": settings.runtime.text_recognition_packing if mode == "recognition" else None}
    def combined(kind: str, values: list[np.ndarray]) -> dict[str, Any]:
        if not values: return {"stage": kind, "status": "unsupported", "input_count": 0}
        began = time.perf_counter(); lines = []; detection_calls = []; detector = models.text_detector()
        for start in range(0, len(values), settings.runtime.text_detection_batch_size):
            chunk = values[start:start + settings.runtime.text_detection_batch_size]; inputs = _pad_detection_batch(list(enumerate(chunk))); call_start = time.perf_counter(); detected = detector.detect_batch([value for _, value in inputs]); detection_calls.append({"submitted_batch_size": len(inputs), "tensor_batch_size": len(inputs), "wall_seconds": time.perf_counter() - call_start})
            for image, result in zip(values[start:start + len(chunk)], detected):
                for region in result.regions:
                    line, _ = _line_crop(image, region.polygon); lines.append(line)
        recognition_calls = []; recognizer = models.text_recognizer()
        for start in range(0, len(lines), settings.runtime.text_recognition_batch_size):
            chunk = lines[start:start + settings.runtime.text_recognition_batch_size]; call_start = time.perf_counter(); recognizer.recognize_batch(chunk); recognition_calls.append({"submitted_batch_size": len(chunk), "tensor_batch_size": len(chunk), "wall_seconds": time.perf_counter() - call_start})
        total = time.perf_counter() - began
        return {"stage": kind, "status": "ok", "input_count": len(values), "detected_lines": len(lines), "recognized_lines": len(lines), "detection_calls": detection_calls, "recognition_calls": recognition_calls, "detection_model_call_count": len(detection_calls), "recognition_model_call_count": len(recognition_calls), "detection_tensor_batch_sizes": [c["tensor_batch_size"] for c in detection_calls], "recognition_tensor_batch_sizes": [c["tensor_batch_size"] for c in recognition_calls], "seconds": total, "throughput_per_second": len(values) / total, "lines_per_second": len(lines) / total, "ms_per_line": total * 1000 / len(lines) if lines else None, "packing": settings.runtime.text_recognition_packing}
    rows = []
    for size in (4, 8, 16) if corpus["localization"] else ():
        for name, localizer in (("docaligner", models.document_localizer()), ("mrz_localizer", models.mrz_localizer())):
            values = corpus["localization"]; rows.append({"stage": name, "mode": "localization", "batch_size": size, "input_count": len(values), "tensor_batch_sizes": [], "status": "ok"})
            localizer_result, diag = _localize(localizer, [(str(i), image) for i, image in enumerate(values)], size, pad=settings.driving_license.aligner_padding if name == "docaligner" else 0); rows[-1].update({"seconds": diag["wall_seconds"], "throughput_per_second": len(values) / diag["wall_seconds"], "ms_per_input": diag["wall_seconds"] * 1000 / len(values), "tensor_batch_sizes": diag["tensor_batch_sizes"], "model_call_count": diag["model_call_count"]})
    for name, values in (("visible_detector", corpus["detection_visible"]), ("mrz_detector", corpus["detection_mrz"])):
        for size in (4, 8, 16): rows.append(bench(name, values, size, "detection"))
    for name, values in (("visible_recognizer", corpus["recognition_visible"]), ("mrz_recognizer", corpus["recognition_mrz"])):
        for size in (8, 16, 32, 64): rows.append(bench(name, values, size, "recognition"))
    rows.extend([combined("visible_combined", corpus["detection_visible"]), combined("mrz_combined", corpus["detection_mrz"])])
    (output / "micro_corpus.json").write_text(json.dumps({"counts": {key: len(value) for key, value in corpus.items()}, "sha256": {key: hashlib.sha256(b"".join(v.tobytes() for v in value)).hexdigest() for key, value in corpus.items()}}, indent=2), encoding="utf-8")
    return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows: path.write_text("", encoding="utf-8"); return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys); writer.writeheader(); writer.writerows({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (dict, list, tuple)) else value for key, value in row.items()} for row in rows)


def plots(output: Path, rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError: return
    good = [row for row in rows if row.get("status") == "ok" and row.get("total_seconds") is not None]
    if not good: return
    figure, axis = plt.subplots(figsize=(10, 6));
    for kind in DOC_TYPES:
        selected = [row for row in good if row["document_type"] == kind]
        axis.plot([row["logical_count"] for row in selected], [row["total_seconds"] for row in selected], "o-", label=kind)
    axis.set(xlabel="logical documents", ylabel="seconds", title="Voight benchmark scaling"); axis.grid(alpha=.3); axis.legend(); figure.tight_layout(); figure.savefig(output / "throughput_scaling.png", dpi=140); plt.close(figure)


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--dataset-root", type=Path, default=ROOT / "dataset"); parser.add_argument("--output-dir", type=Path); parser.add_argument("--base-url", default="http://127.0.0.1:8000"); parser.add_argument("--timeout", type=float, default=900); parser.add_argument("--repeats", type=int, default=3); parser.add_argument("--document-type", choices=DOC_TYPES, required=True); parser.add_argument("--variant", choices=ALL_VARIANTS); parser.add_argument("--sensitivity-only", action="store_true"); parser.add_argument("--skip-production", action="store_true"); parser.add_argument("--skip-sensitivity", action="store_true"); parser.add_argument("--scaling", type=int, nargs="+", default=[1, 2, 4, 8, 16]); return parser.parse_args()


def main() -> int:
    cli = args()
    if cli.repeats < 3 or any(n < 1 for n in cli.scaling): raise SystemExit("use at least three measured repeats and positive scaling sizes")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"); output = cli.output_dir or ROOT / "outputs" / "benchmarks" / "pipeline_breakdown" / stamp; output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc).isoformat(); documents, manifest = validate_and_manifest(cli.dataset_root); documents = [document for document in documents if document.document_type == cli.document_type]; manifest["selected_document_type"] = cli.document_type; manifest["selected_documents"] = [document["document_id"] for document in manifest["documents"] if document["document_type"] == cli.document_type]; settings = _settings(); (output / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8"); (output / "environment.json").write_text(json.dumps(environment(settings, manifest, started), indent=2), encoding="utf-8")
    needs_models = cli.sensitivity_only or not cli.variant or not cli.variant.endswith("_full_production")
    models = Models(settings) if needs_models else None; print(f"dataset: {manifest['counts']} output: {output}")
    if models is not None:
        if cli.sensitivity_only:
            loaders = (models.document_localizer, models.mrz_localizer, models.text_detector, models.text_recognizer)
        else:
            scope = variant_scope(cli.document_type, cli.variant or "")
            loaders = []
            if cli.document_type == "driving_license" or scope["visible_ocr"]:
                loaders.append(models.document_localizer)
            if cli.document_type != "driving_license" and (cli.document_type == "passport" or scope["mrz_ocr"] or cli.variant == "id_card_visible_probe"):
                loaders.append(models.mrz_localizer)
            if scope["visible_ocr"] or scope["mrz_ocr"]:
                loaders.extend((models.text_detector, models.text_recognizer))
        for loader in loaders:
            loader()
    all_runs: list[Run] = []
    if not cli.skip_production and not cli.sensitivity_only:
        requests.get(f"{cli.base_url.rstrip('/')}/v1/health/ready", timeout=30).raise_for_status()
        for kind in (cli.document_type,):
            full = [d for d in documents if d.document_type == kind]; warm = run_production(settings, full[:1], kind, 0, cli.base_url, cli.timeout); print(f"warmup {kind}: {warm.status}")
            for repeat in range(1, cli.repeats + 1):
                all_runs.append(run_production(settings, full, kind, repeat, cli.base_url, cli.timeout))
            for n in cli.scaling:
                for repeat in range(1, cli.repeats + 1): all_runs.append(run_production(settings, cycles(documents, kind, n), kind, repeat, cli.base_url, cli.timeout))
    # Direct ablations use deterministic rotation so the first measured variant changes by repeat.
    names = [] if cli.sensitivity_only else ([cli.variant] if cli.variant else VARIANTS[cli.document_type])
    for kind, names in ((cli.document_type, names),):
        for n in cli.scaling:
            source = cycles(documents, kind, n)
            for repeat in range(1, cli.repeats + 1):
                for offset in range(len(names)):
                    variant = names[(offset + repeat - 1) % len(names)]
                    if variant.endswith("_full_production"):
                        continue
                    if n != cli.scaling[0] and variant.endswith("side_visible_breakdown"): continue
                    try:
                        total, diagnostics, outputs = run_partial(settings, models, source, kind, variant)
                        all_runs.append(Run(variant, kind, repeat, n, consumed_physical_count(kind, variant, source, diagnostics), tuple(d.document_id for d in source), "ok", total, None, None, _stage_values(diagnostics, total), diagnostics, outputs))
                    except Exception as error:
                        all_runs.append(Run(variant, kind, repeat, n, sum(d.physical_count for d in source), tuple(d.document_id for d in source), "failed", 0.0, None, None, {}, {}, {}, f"{type(error).__name__}: {error}"))
    sensitivity = [] if cli.skip_sensitivity or not cli.sensitivity_only else microbench(models, settings, documents, output)
    correctness = score(documents, all_runs)
    correctness["output_stability"] = stability(all_runs)
    raw = []
    for run in all_runs:
        row = {"variant": run.variant, "document_type": run.document_type, "repeat": run.repeat, "logical_count": run.logical_count, "physical_count": run.physical_count, "source_ids": run.source_ids, "status": run.status, "total_seconds": run.total_seconds, "client_seconds": run.client_seconds, "server_seconds": run.server_seconds, "stages": run.stages, "diagnostics": _counts(run.diagnostics), "batch_details": _batch_details(run.diagnostics), "output_digests": {key: _digest(value) for key, value in run.outputs.items()}, "error": run.error}
        raw.append(row)
    (output / "raw_measurements.jsonl").write_text("\n".join(json.dumps(row, default=_json_safe) for row in raw) + "\n", encoding="utf-8"); write_rows(output / "raw_measurements.csv", raw); write_rows(output / "batch_sensitivity.csv", sensitivity)
    summary = {"run_count": len(all_runs), "repeats": cli.repeats, "scaling": cli.scaling, "stats": {}}
    for variant in sorted({run.variant for run in all_runs}):
        selected = [run for run in all_runs if run.variant == variant and run.status == "ok"]; summary["stats"][variant] = {"total_seconds": stats(run.total_seconds for run in selected), "by_logical_count": {str(n): {"total_seconds": stats(run.total_seconds for run in selected if run.logical_count == n), "stages": {stage: stats(run.stages.get(stage, 0.0) for run in selected if run.logical_count == n) for stage in (selected[0].stages if selected else ())}} for n in sorted({run.logical_count for run in selected})}}
    analysis = {"amdahl_upper_bound": {}, "measured_deltas": {}}
    for variant, data in summary["stats"].items():
        for n, value in data["by_logical_count"].items():
            total = value["total_seconds"].get("median") or 0.0
            analysis["amdahl_upper_bound"][f"{variant}:{n}"] = {stage: total / (total - info.get("median")) if total > (info.get("median") or 0.0) else None for stage, info in value["stages"].items()}
    for kind in DOC_TYPES:
        for n in cli.scaling:
            rows = {row.variant: row for row in all_runs if row.document_type == kind and row.logical_count == n and row.status == "ok"}
            base = rows.get(f"{kind}_full_production")
            if base:
                for variant, row in rows.items():
                    if variant != base.variant: analysis["measured_deltas"][f"{base.variant}_vs_{variant}:{n}"] = {"full_median_seconds": base.total_seconds, "variant_median_seconds": row.total_seconds, "delta_seconds": base.total_seconds - row.total_seconds}
    (output / "analysis.json").write_text(json.dumps(analysis, indent=2, default=_json_safe), encoding="utf-8"); (output / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_safe), encoding="utf-8"); write_rows(output / "summary.csv", [{"variant": variant, "logical_count": n, "total_seconds": value["total_seconds"].get("median"), **{f"stage_{stage}": info.get("median") for stage, info in value["stages"].items()}} for variant, data in summary["stats"].items() for n, value in data["by_logical_count"].items()]); (output / "correctness.json").write_text(json.dumps(correctness, indent=2, default=_json_safe), encoding="utf-8")
    final_profile = {
        "pipeline": "FINAL Latin",
        "runtime": environment(settings, manifest, started),
        "measurement_notes": {
            "seconds": "wall-clock stage time; medians over measured repeats",
            "recognition_crops": "actual line-crop dimensions passed to recognition, grouped by model call",
            "detection_tensor_sizes": "actual batch N submitted to Paddle; Paddle's internal resized H/W is not exposed by PaddleOCR",
            "process_peak_rss_mb": "Linux ru_maxrss for the API process, including loaded models",
        },
        "documents": {},
    }
    for kind in DOC_TYPES:
        count = len([document for document in documents if document.document_type == kind])
        selected = [run for run in all_runs if run.variant == f"{kind}_full_production" and run.logical_count == count and run.status == "ok"]
        if not selected:
            continue
        rss = [value for value in (_batch_details(run.diagnostics).get("process_peak_rss_mb") for run in selected) if value is not None]
        final_profile["documents"][kind] = {
            "logical_documents": count,
            "physical_images": sum(document.physical_count for document in documents if document.document_type == kind),
            "repeats": len(selected),
            "stages_seconds": {stage: stats(run.stages.get(stage, 0.0) for run in selected) for stage in selected[0].stages},
            "total_seconds": stats(run.total_seconds for run in selected),
            "process_peak_rss_mb": stats(rss),
            "batch_details": _batch_details(selected[0].diagnostics),
        }
    (output / "final_latin_profile.json").write_text(json.dumps(final_profile, indent=2, default=_json_safe), encoding="utf-8")
    write_rows(output / "stage_breakdown.csv", [{"variant": run.variant, "document_type": run.document_type, "repeat": run.repeat, "logical_count": run.logical_count, "stage": stage, "seconds": seconds, "share": seconds / run.total_seconds if run.total_seconds else None} for run in all_runs for stage, seconds in run.stages.items()]); plots(output, raw)
    (output / "run_metadata.json").write_text(json.dumps({"benchmark_finished_utc": datetime.now(timezone.utc).isoformat(), "model_load_seconds": models._load_seconds if models is not None else {}}, indent=2), encoding="utf-8")
    if models is not None:
        models.close(); del models
    gc.collect(); print(f"completed {len(all_runs)} measured runs; results: {output}"); return 0 if all(run.status == "ok" for run in all_runs) else 2


if __name__ == "__main__": raise SystemExit(main())
