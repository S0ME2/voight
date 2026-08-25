"""Compare generic Paddle MRZ OCR with specialized MRZScanner on the same crops."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--passport", type=Path, default=ROOT / "annotation_input/passports/passport.png")
    parser.add_argument("--id-back", type=Path, default=ROOT / "annotation_input/id_cards/uzbekistan_id_001/back.png")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def valid(text: str, kind: str) -> bool:
    from app.contracts import ValidationStatus
    from app.documents.mrz import parse

    result = parse(text, kind)
    return bool(result.raw_lines) and all(item.status != ValidationStatus.FAILED for item in result.validations)


def main() -> int:
    args = arguments()
    if not args.passport.is_file() or not args.id_back.is_file():
        print("MRZ benchmark inputs are missing", file=sys.stderr)
        return 2

    from app.config import Settings
    from app.documents.mrz import ID_CARD, PASSPORT, crop_polygon, preprocess, reconstruct, select
    from app.inference.batch import BatchedOcr, OcrSample
    from app.models import Models

    settings = Settings.from_env()
    if settings.runtime.target != "cpu" or settings.runtime.text_recognition_processes != 1:
        print("set RUNTIME_TARGET=cpu and TEXT_RECOGNITION_PROCESSES=1", file=sys.stderr)
        return 2
    images = [cv2.imread(str(args.passport)), cv2.imread(str(args.id_back))]
    if any(image is None for image in images):
        print("MRZ benchmark inputs could not be decoded", file=sys.stderr)
        return 2

    models = Models(settings)
    localizer = models.mrz_localizer()
    localized = localizer.localize_batch(images)
    crops = [
        crop_polygon(image, result.polygon, settings.mrz.polygon_padding_ratio)[0]
        for image, result in zip(images, localized)
    ]
    processed_crops = [
        preprocess(
            crop,
            settings.mrz.max_side,
            settings.mrz.contrast,
        )
        for crop in crops
    ]
    profiles = [PASSPORT, ID_CARD]
    kinds = ["passport", "id_card"]

    generic = BatchedOcr(
        models.text_detector(), models.text_recognizer(),
        detection_batch_size=settings.runtime.text_detection_batch_size,
        recognition_batch_size=settings.runtime.text_recognition_batch_size,
    )
    started = time.perf_counter()
    old = generic.run([OcrSample(str(index), crop) for index, crop in enumerate(processed_crops)])
    old_seconds = time.perf_counter() - started
    old_lines = []
    for index, profile in enumerate(profiles):
        selected = select(reconstruct(old.tokens.get(str(index), [])), profile.line_counts)
        old_lines.append(tuple(line.text for line in selected))

    specialized_settings = replace(
        settings,
        models=replace(
            settings.models,
            mrz=replace(settings.models.mrz, recognizer_backend="mrzscanner"),
        ),
    )
    recognizer = Models(specialized_settings).mrz_recognizer()
    started = time.perf_counter()
    new = recognizer.recognize_batch(crops)
    new_seconds = time.perf_counter() - started
    new_lines = [result.lines for result in new]
    rows = []
    for index, kind in enumerate(kinds):
        old_text, new_text = "\n".join(old_lines[index]), "\n".join(new_lines[index])
        rows.append({
            "document_type": kind,
            "expected_mrz_available": False,
            "exact_full_mrz_match_against_truth": None,
            "per_line_exact_match_against_truth": None,
            "character_accuracy": None,
            "old_lines": old_lines[index],
            "new_lines": new_lines[index],
            "old_parser_valid": valid(old_text, kind),
            "new_parser_valid": valid(new_text, kind),
            "old_new_full_match": old_text == new_text,
            "old_new_line_matches": sum(left == right for left, right in zip(old_lines[index], new_lines[index])),
        })
    report = {
        "scope": "same detected MRZ crops; no labeled MRZ truth is available in evaluation_ground_truth.json",
        "old": {
            "backend": "generic-paddle",
            "seconds": old_seconds,
            "documents_per_second": len(crops) / old_seconds,
            "detection_batch_sizes": old.diagnostics["text_detection"]["tensor_batch_sizes"],
            "recognition_batch_sizes": old.diagnostics["text_recognition"]["tensor_batch_sizes"],
            "failure_count": len(old.errors),
            "failure_rate": len(old.errors) / len(crops),
        },
        "new": {
            "backend": "mrzscanner",
            "model": settings.models.mrz.recognizer_model,
            "seconds": new_seconds,
            "documents_per_second": len(crops) / new_seconds,
            "recognition_batch_sizes": recognizer.last_tensor_batch_sizes,
            "model_supports_batch": recognizer.supports_batch,
            "failure_count": sum(result.status != "recognized" for result in new),
            "failure_rate": sum(result.status != "recognized" for result in new) / len(crops),
        },
        "rows": rows,
    }
    output = args.output or ROOT / "benchmarks/results" / f"mrz-{datetime.now():%Y%m%dT%H%M%S}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
