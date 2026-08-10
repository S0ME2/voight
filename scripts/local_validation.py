"""CPU-only, repeatable evidence for the current profile OCR pipeline.

This uses the three annotated images with controlled OCR tokens.  It validates
profile geometry and batch plumbing, not population OCR accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.artifacts import ArtifactWriter
from app.documents.identity import extract_id_card, extract_passport
from app.inference.batch import BatchedOcr, OcrSample


ROOT = Path(__file__).resolve().parents[1]
TRUTH = json.loads((ROOT / "annotations/evaluation_ground_truth.json").read_text())["samples"]
PASSPORT_PROFILE = ROOT / "config/documents/uz_passport/profile.json"
ID_PROFILE = ROOT / "config/documents/uz_id_card/profile.json"
WRITER = ArtifactWriter(Path("/unused"), "", "", False)


def _tokens(profile_path: Path, region: str, values: dict[str, str], crop: np.ndarray) -> list[dict[str, Any]]:
    profile = json.loads(profile_path.read_text())
    height, width = crop.shape[:2]
    result = []
    for index, (field, value) in enumerate(values.items()):
        roi = profile["regions"][region]["field_rois"][field]
        x1, y1, x2, y2 = roi["x1"] * width, roi["y1"] * height, roi["x2"] * width, roi["y2"] * height
        result.append({"index": index, "text": value, "score": 0.9, "x1": x1 + 1, "y1": y1 + 1, "x2": x2 - 1, "y2": y2 - 1, "center_x": (x1 + x2) / 2, "center_y": (y1 + y2) / 2, "height": max(1, y2 - y1 - 2)})
    return result


def _controlled_values(profile_path: Path, region: str) -> dict[str, str]:
    """Keep geometry validation independent from optional annotation truth text."""
    profile = json.loads(profile_path.read_text())
    return {name: f"VALUE_{name.upper()}" for name in profile["regions"][region]["field_rois"]}


def _corners(sample: dict[str, Any], transform: np.ndarray | None = None) -> dict[str, Any]:
    size = sample["original_size"]
    points = np.asarray([[x * size["width"], y * size["height"]] for x, y in sample["corners"]], dtype=np.float32)
    if transform is not None:
        points = cv2.perspectiveTransform(points[None, :, :], transform)[0]
    return {"corners": points.tolist(), "score": 0.91}


def _transform(image: np.ndarray, kind: str) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    identity = np.eye(3, dtype=np.float32)
    if kind == "rotation":
        affine = cv2.getRotationMatrix2D((width / 2, height / 2), 3, 1)
        return cv2.warpAffine(image, affine, (width, height), borderValue=(32, 32, 32)), np.vstack((affine, (0, 0, 1))).astype(np.float32)
    if kind == "perspective":
        source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
        destination = np.float32([[8, 5], [width - 12, 2], [width - 3, height - 8], [4, height - 2]])
        matrix = cv2.getPerspectiveTransform(source, destination)
        return cv2.warpPerspective(image, matrix, (width, height), borderValue=(32, 32, 32)), matrix
    if kind == "background":
        canvas = np.full_like(image, (48, 72, 96))
        mask = np.full(image.shape[:2], 255, dtype=np.uint8)
        return cv2.copyTo(image, mask, canvas), identity
    if kind == "blur":
        return cv2.GaussianBlur(image, (5, 5), 0), identity
    if kind == "glare":
        glare = image.copy()
        cv2.ellipse(glare, (width * 3 // 4, height // 4), (max(8, width // 8), max(8, height // 10)), 0, 0, 360, (255, 255, 255), -1)
        return cv2.addWeighted(image, 0.72, glare, 0.28, 0), identity
    raise ValueError(f"Unknown transform: {kind}")


def _passport_result(image: np.ndarray, transform: np.ndarray | None) -> Any:
    sample = TRUTH["passport:passport.png"]
    expected = _controlled_values(PASSPORT_PROFILE, "data_page")
    return extract_passport(image, PASSPORT_PROFILE, lambda _image: _corners(sample, transform), lambda crop: _tokens(PASSPORT_PROFILE, "data_page", expected, crop), lambda _image: sample["mrz"], WRITER)


def _id_result(front: np.ndarray, back: np.ndarray, front_transform: np.ndarray | None, back_transform: np.ndarray | None) -> Any:
    front_sample = TRUTH["id_card:uzbekistan_id_001:front"]
    back_sample = TRUTH["id_card:uzbekistan_id_001:back"]
    detections = iter((_corners(front_sample, front_transform), _corners(back_sample, back_transform)))
    recognitions = iter((("front", _controlled_values(ID_PROFILE, "front")), ("back", _controlled_values(ID_PROFILE, "back"))))
    return extract_id_card(front, back, ID_PROFILE, lambda _image: next(detections), lambda crop: _tokens(ID_PROFILE, *next(recognitions), crop), lambda image: back_sample["mrz"] if image is back else "", WRITER)


def _field_report(result: Any, expected: dict[str, str]) -> dict[str, Any]:
    actual = {name: field.value for name, field in result.fields.items()}
    matched = sorted(name for name, value in expected.items() if actual.get(name) == value)
    missing = sorted(name for name in expected if not actual.get(name))
    return {"configured_fields": len(expected), "exact_matches": len(matched), "matched_fields": matched, "missing_fields": missing, "mrz_valid": all(item.status.value == "passed" for item in result.mrz.validations), "confidence_sources": sorted({field.confidence.source.value for field in result.fields.values() if field.confidence})}


def extraction_evidence() -> dict[str, Any]:
    passport_image = cv2.imread(str(ROOT / TRUTH["passport:passport.png"]["source"]))
    front_image = cv2.imread(str(ROOT / TRUTH["id_card:uzbekistan_id_001:front"]["source"]))
    back_image = cv2.imread(str(ROOT / TRUTH["id_card:uzbekistan_id_001:back"]["source"]))
    passport_expected = _controlled_values(PASSPORT_PROFILE, "data_page")
    passport = _field_report(_passport_result(passport_image, None), passport_expected)
    identity_expected = {**_controlled_values(ID_PROFILE, "front"), **_controlled_values(ID_PROFILE, "back")}
    identity = _field_report(_id_result(front_image, back_image, None, None), identity_expected)
    transforms = {}
    for kind in ("rotation", "perspective", "background", "blur", "glare"):
        passport_variant, passport_matrix = _transform(passport_image, kind)
        front_variant, front_matrix = _transform(front_image, kind)
        back_variant, back_matrix = _transform(back_image, kind)
        transforms[kind] = {"passport": _field_report(_passport_result(passport_variant, passport_matrix), passport_expected), "id_card": _field_report(_id_result(front_variant, back_variant, front_matrix, back_matrix), identity_expected)}
    return {"scope": "one annotated passport and one annotated ID-card pair; controlled OCR tokens exercise profile extraction only", "baseline": {"passport": passport, "id_card": identity}, "synthetic_robustness": transforms}


class _Detector:
    def predict(self, *, input: list[np.ndarray], batch_size: int) -> list[dict[str, Any]]:
        images = input
        assert batch_size == len(images)
        polygon = np.float32([[5, 5], [35, 5], [35, 20], [5, 20]])
        return [{"dt_polys": [polygon]} for _ in images]


class _Recognizer:
    def predict(self, *, input: list[np.ndarray], batch_size: int) -> list[dict[str, Any]]:
        images = input
        assert batch_size == len(images)
        return [{"rec_text": "OK", "rec_score": 0.9} for _ in images]


def _measure(batch_size: int, repeats: int) -> dict[str, Any]:
    samples = [OcrSample(str(index), np.zeros((40, 40, 3), dtype=np.uint8)) for index in range(batch_size)]
    ocr = BatchedOcr(_Detector(), _Recognizer(), detection_batch_size=batch_size, recognition_batch_size=batch_size)
    started = time.perf_counter()
    result = None
    for _ in range(repeats):
        result = ocr.run(samples)
    seconds = time.perf_counter() - started
    return {"batch_size": batch_size, "repeats": repeats, "seconds": seconds, "throughput_samples_per_second": batch_size * repeats / seconds, "model_call_proof": {stage: result.diagnostics[stage]["tensor_batch_sizes"] for stage in ("text_detection", "text_recognition")}, "max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}


def batch_evidence(sizes: list[int], repeats: int) -> dict[str, Any]:
    rows = []
    for size in sizes:
        single = _measure(1, repeats * size)
        batched = _measure(size, repeats)
        rows.append({"size": size, "single": single, "true_batch": batched})
    return {"scope": "synthetic detector/recognizer timing; use only for batch call and resource baselines", "rows": rows, "true_batch_observed": any(max(row["true_batch"]["model_call_proof"]["text_detection"]) > 1 for row in rows if row["size"] > 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a CPU-only local validation baseline.")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/local-validation-baseline.json")
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or any(size < 1 for size in args.sizes):
        raise SystemExit("--repeats and --sizes must be positive")
    report = {"schema_version": 1, "runtime": {"target": "cpu", "pid": os.getpid()}, "extraction": extraction_evidence(), "batch": batch_evidence(args.sizes, args.repeats), "limitations": ["No population accuracy percentage is reported.", "Synthetic robustness preserves annotated geometry and supplies controlled OCR tokens.", "Confidence sources are uncalibrated OCR-token means."]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()
