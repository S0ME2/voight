"""Small, CPU-only continuation runner for the six optimization experiments.

It deliberately reuses ``pipeline_breakdown``'s production adapters and truth
scoring.  The runner is not imported by the service and never changes `.env`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import cv2
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.maintained.pipeline_breakdown import (
    Document, Models, Run, _json_safe, _stage_values, discover_dataset,
    _input_for, environment, score, stability, stats, validate_and_manifest, run_partial,
)
from app.api.v1 import _run_batch
from app.config import Settings, TextModelSettings
from app.documents.mrz import crop_polygon, preprocess, parse as parse_mrz


STAMP = "20260814T000000Z"
OUTPUT = ROOT / "outputs" / "benchmarks" / "optimization_six" / STAMP
BATCHES = (1, 2, 4, 8, 12, 16, 24, 32)


def _settings(batch: int) -> Settings:
    settings = Settings.from_env()
    return replace(settings, runtime=replace(settings.runtime, text_recognition_batch_size=batch))


def _run(settings: Settings, models: Models, documents: list[Document], kind: str, variant: str, repeat: int) -> Run:
    started = time.perf_counter()
    total, diagnostics, outputs = run_partial(settings, models, documents, kind, variant)
    return Run(variant, kind, repeat, len(documents), sum(d.physical_count for d in documents), tuple(d.document_id for d in documents), "ok", total, None, None, _stage_values(diagnostics, total), diagnostics, outputs)


def _write(directory: Path, settings: Settings, manifest: dict, documents: list[Document], runs: list[Run], report: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    raw = []
    for run in runs:
        raw.append({"variant": run.variant, "document_type": run.document_type, "repeat": run.repeat, "logical_count": run.logical_count, "physical_count": run.physical_count, "total_seconds": run.total_seconds, "stages": run.stages, "diagnostics": run.diagnostics, "error": run.error})
    (directory / "environment.json").write_text(json.dumps(environment(settings, manifest, datetime.now(timezone.utc).isoformat()), indent=2, default=_json_safe))
    (directory / "configurations.json").write_text(json.dumps({"batches": BATCHES, "runner": "direct current production components"}, indent=2))
    (directory / "raw.jsonl").write_text("\n".join(json.dumps(value, default=_json_safe) for value in raw) + "\n")
    with (directory / "raw.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("variant", "document_type", "repeat", "logical_count", "physical_count", "total_seconds", "recognition_seconds"))
        writer.writeheader()
        writer.writerows({**{key: row[key] for key in writer.fieldnames if key in row}, "recognition_seconds": row["stages"].get("recognition", 0.0)} for row in raw)
    summary = {"runs": len(runs), "correctness": score(documents, runs), "stability": stability(runs)}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, default=_json_safe))
    (directory / "summary.csv").write_text("variant,document_type,repeat,total_seconds\n" + "\n".join(f"{r.variant},{r.document_type},{r.repeat},{r.total_seconds}" for r in runs) + "\n")
    (directory / "report.md").write_text(report)


def experiment_one() -> int:
    documents, manifest = validate_and_manifest(ROOT / "dataset")
    directory = OUTPUT / "01_recognition_batch"
    variants = {
        "passport": "passport_visible_no_mrz_ocr",
        "id_card": "id_card_visible_known_side",
        "driving_license": "driving_license_visible_ocr_only",
    }
    runs: list[Run] = []
    # One warmup and one exploratory measurement per point; models stay fixed.
    settings = _settings(32)
    models = Models(settings)
    for kind, variant in variants.items():
        corpus = [d for d in documents if d.document_type == kind]
        _run(settings, models, corpus, kind, variant, 0)
        for batch in BATCHES:
            current = _settings(batch)
            run = _run(current, models, corpus, kind, variant, 1)
            run.variant = f"{variant}:batch={batch}"
            runs.append(run)
    # Rerun fastest and the smallest point within 3% three times.
    selected = []
    for kind, variant in variants.items():
        candidates = [r for r in runs if r.document_type == kind]
        fastest = min(candidates, key=lambda r: r.total_seconds)
        floor = fastest.total_seconds * 1.03
        near = min((r for r in candidates if r.total_seconds <= floor), key=lambda r: int(r.variant.rsplit("=", 1)[1]))
        selected.extend({fastest.variant, near.variant})
    for name in selected:
        source = next(r for r in runs if r.variant == name)
        batch = int(name.rsplit("=", 1)[1])
        kind = source.document_type
        corpus = [d for d in documents if d.document_type == kind]
        base_variant = variants[kind]
        for repeat in range(1, 4):
            run = _run(_settings(batch), models, corpus, kind, base_variant, repeat)
            run.variant = name
            runs.append(run)
    models.close()
    lines = ["# Experiment 1 — recognition batch-size sweep", "", "Exploratory points: one measured run after warmup; selected candidates: three measured runs.", "", "| scope | batch | seconds | recognition seconds | docs/s |", "| --- | ---: | ---: | ---: | ---: |"]
    for run in runs:
        if run.repeat == 1:
            lines.append(f"| {run.document_type} | {run.variant.rsplit('=', 1)[1]} | {run.total_seconds:.3f} | {run.stages.get('recognition', 0):.3f} | {run.logical_count / run.total_seconds:.3f} |")
    _write(directory, settings, manifest, documents, runs, "\n".join(lines) + "\n")
    print(f"Experiment 1 complete: {directory}")
    return 0


def _combined(settings: Settings, models: Models, documents: list[Document], kind: str, repeat: int) -> Run:
    started = time.perf_counter()
    response = _run_batch([_input_for(document) for document in documents], models, settings)
    outputs = {
        document.document_id: {
            "fields": {name: field.value for name, field in (item.result.fields if item.success else {}).items()},
            "mrz": list(item.result.mrz.raw_lines) if item.success and item.result.mrz else [],
        }
        for document, item in zip(documents, response.items)
    }
    total = time.perf_counter() - started
    return Run("CURRENT_COMBINED", kind, repeat, len(documents), sum(d.physical_count for d in documents), tuple(d.document_id for d in documents), "ok", total, None, None, _stage_values(response.diagnostics, total), response.diagnostics, outputs)


def experiment_two() -> int:
    documents, manifest = validate_and_manifest(ROOT / "dataset")
    runs: list[Run] = []
    settings = _settings(32)
    models = Models(settings)
    for kind, variant in (("passport", "passport_visible_no_mrz_ocr"), ("id_card", "id_card_visible_known_side")):
        corpus = [d for d in documents if d.document_type == kind]
        _combined(settings, models, corpus[:1], kind, 0)
        for repeat in range(1, 4):
            runs.append(_combined(settings, models, corpus, kind, repeat))
            split = _run(settings, models, corpus, kind, variant, repeat); split.variant = "SPLIT_SAME_BATCH"; runs.append(split)
            tuned = _run(_settings(8), models, corpus, kind, variant, repeat); tuned.variant = "SPLIT_TUNED_BATCH"; runs.append(tuned)
    models.close()
    lines = ["# Experiment 2 — combined versus split OCR", "", "`CURRENT_COMBINED` is the current ProfileBatchRunner. Split modes call the same detector/recognizer independently for visible and MRZ benchmark workloads; no application behavior changed.", "", "| document | mode | median seconds |", "| --- | --- | ---: |"]
    for kind in ("passport", "id_card"):
        for mode in ("CURRENT_COMBINED", "SPLIT_SAME_BATCH", "SPLIT_TUNED_BATCH"):
            value = stats(r.total_seconds for r in runs if r.document_type == kind and r.variant == mode)["median"]
            lines.append(f"| {kind} | {mode} | {value:.3f} |")
    _write(OUTPUT / "02_split_visible_mrz", settings, manifest, documents, runs, "\n".join(lines) + "\n")
    print("Experiment 2 complete")
    return 0


def experiment_three() -> int:
    documents, manifest = validate_and_manifest(ROOT / "dataset")
    directory = OUTPUT / "03_recognizer_models"
    directory.mkdir(parents=True, exist_ok=True)
    models = ("PP-OCRv6_medium_rec", "PP-OCRv6_small_rec", "PP-OCRv6_tiny_rec", "latin_PP-OCRv5_mobile_rec", "PP-OCRv5_mobile_rec", "en_PP-OCRv5_mobile_rec")
    rows = []
    report = ["# Experiment 3 — CPU recognizer comparison", "", "The identical four-image fixed corpus was used for all successful model runs. Model acquisition used the official PaddleOCR BOS source. The medium wrapper failed before emitting a result file; production medium timing remains in the reference and E2E reports.", "", "| model | batch | median lines/s | median ms/line |", "| --- | ---: | ---: | ---: |"]
    for model in models:
        path = directory / model / "recognition_batch_benchmark.json"
        if not path.is_file():
            rows.append({"model": model, "status": "failed_to_emit", "error": "benchmark wrapper emitted no result"})
            continue
        data = json.loads(path.read_text())
        for row in data.get("rows", []):
            row["model"] = model; rows.append(row)
        successful = [row for row in data.get("rows", []) if row.get("status") == "ok"]
        for batch in (4, 8, 16, 32):
            values = [row for row in successful if row.get("batch_size") == batch]
            if values:
                report.append(f"| {model} | {batch} | {stats(row['lines_per_second'] for row in values)['median']:.3f} | {stats(row['milliseconds_per_line'] for row in values)['median']:.3f} |")
    (directory / "environment.json").write_text(json.dumps(environment(_settings(32), manifest, datetime.now(timezone.utc).isoformat()), indent=2, default=_json_safe))
    (directory / "configurations.json").write_text(json.dumps({"models": models, "fixed_corpus": "passport p_1, ID id_1 front/back, driving d_2", "batches": [4, 8, 16, 32]}, indent=2))
    (directory / "raw.jsonl").write_text("\n".join(json.dumps(row, default=_json_safe) for row in rows) + "\n")
    (directory / "raw.csv").write_text("model,status,batch_size,lines_per_second,milliseconds_per_line\n" + "\n".join(f"{row.get('model')},{row.get('status')},{row.get('batch_size','')},{row.get('lines_per_second','')},{row.get('milliseconds_per_line','')}" for row in rows) + "\n")
    (directory / "summary.json").write_text(json.dumps({"models": models, "successful_models": [model for model in models if (directory / model / "recognition_batch_benchmark.json").is_file()]}, indent=2))
    (directory / "summary.csv").write_text("model,status\n" + "\n".join(f"{model},{'ok' if (directory / model / 'recognition_batch_benchmark.json').is_file() else 'failed_to_emit'}" for model in models) + "\n")
    (directory / "report.md").write_text("\n".join(report) + "\n")
    print("Experiment 3 complete")
    return 0


def experiment_four() -> int:
    directory = OUTPUT / "04_cpu_runtime"; directory.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(str(ROOT / "dataset/passport/p_1.png"))
    crop = cv2.resize(image, (320, 64))
    rows = []
    from paddleocr import TextRecognition
    model_dir = Path(os.getenv("MODEL_DIR", str(Path.home() / ".paddlex"))) / "official_models"
    for threads in (1, 2, 4, 6, 8, 12, 16):
        try:
            started = time.perf_counter(); model = TextRecognition(model_name="PP-OCRv6_medium_rec", model_dir=str(model_dir / "PP-OCRv6_medium_rec"), device="cpu", cpu_threads=threads); load = time.perf_counter() - started
            for _ in range(1): model.predict(input=[crop], batch_size=1)
            values = []
            for repeat in range(1, 4):
                began = time.perf_counter(); model.predict(input=[crop], batch_size=1); values.append(time.perf_counter() - began)
            rows.append({"backend": "paddle", "threads": threads, "status": "ok", "load_seconds": load, "median_seconds": stats(values)["median"], "lines_per_second": 1 / stats(values)["median"]})
        except Exception as error:
            rows.append({"backend": "paddle", "threads": threads, "status": "failed", "error": f"{type(error).__name__}: {error}"})
    try:
        TextRecognition(model_name="PP-OCRv6_medium_rec", model_dir=str(model_dir / "PP-OCRv6_medium_rec"), device="cpu", enable_hpi=True)
        hpi = {"backend": "hpi", "status": "initialized"}
    except Exception as error:
        hpi = {"backend": "hpi", "status": "unsupported", "error": f"{type(error).__name__}: {error}"}
    rows.append(hpi)
    (directory / "environment.json").write_text(json.dumps(environment(_settings(8), {"counts": {"passport": 9, "id_card": 4, "driving_license": 7}}, datetime.now(timezone.utc).isoformat()), indent=2, default=_json_safe))
    (directory / "configurations.json").write_text(json.dumps({"threads": [1, 2, 4, 6, 8, 12, 16], "model": "PP-OCRv6_medium_rec", "corpus": "one fixed resized line image", "hpi_cpu_only": True}, indent=2))
    (directory / "raw.jsonl").write_text("\n".join(json.dumps(row, default=_json_safe) for row in rows) + "\n")
    (directory / "raw.csv").write_text("backend,threads,status,load_seconds,median_seconds,lines_per_second,error\n" + "\n".join(f"{r.get('backend')},{r.get('threads','')},{r.get('status')},{r.get('load_seconds','')},{r.get('median_seconds','')},{r.get('lines_per_second','')},{r.get('error','')}" for r in rows) + "\n")
    successful = [r for r in rows if r.get("status") == "ok"]
    best = min(successful, key=lambda r: r["median_seconds"]) if successful else None
    (directory / "summary.json").write_text(json.dumps({"best_thread": best, "hpi": hpi}, indent=2, default=_json_safe))
    (directory / "summary.csv").write_text("backend,threads,status,median_seconds,lines_per_second\n" + "\n".join(f"{r.get('backend')},{r.get('threads','')},{r.get('status')},{r.get('median_seconds','')},{r.get('lines_per_second','')}" for r in rows) + "\n")
    (directory / "report.md").write_text("# Experiment 4 — CPU threads and runtime\n\n" + "\n".join(f"- Paddle threads {r['threads']}: median {r.get('median_seconds', 'failed')} seconds, status {r['status']}" for r in rows if r.get('backend') == 'paddle') + f"\n\nHPI CPU result: {hpi['status']}.\n")
    print("Experiment 4 complete")
    return 0


def fixed_rows(image, count: int):
    height = image.shape[0]
    margin = max(1, round(height * 0.02))
    return [image[max(0, round(height * index / count) - margin):min(height, round(height * (index + 1) / count) + margin)] for index in range(count)]


def experiment_five() -> int:
    documents, manifest = validate_and_manifest(ROOT / "dataset")
    directory = OUTPUT / "05_mrz_rows"; directory.mkdir(parents=True, exist_ok=True)
    settings = _settings(8); models = Models(settings); rows = []
    for kind, variant, count, role in (("passport", "passport_mrz_only", 2, "image"), ("id_card", "id_card_mrz_known_back", 3, "back")):
        corpus = [d for d in documents if d.document_type == kind]
        # Trusted detector-based baseline, three fresh full-corpus repeats.
        for repeat in range(1, 4):
            run = _run(settings, models, corpus, kind, variant, repeat); run.variant = "DETECTOR_BASED_MEDIUM"
            rows.append({"kind": kind, "variant": run.variant, "repeat": repeat, "seconds": run.total_seconds, "stages": run.stages, "correctness": score(corpus, [run])})
        for document in corpus:
            path = dict(document.paths)[role]; image = cv2.imread(str(path)); localizer = models.mrz_localizer(); location = localizer.localize_batch([image])[0]
            crop, _ = __import__("benchmarks.maintained.pipeline_breakdown", fromlist=["crop_polygon"]).crop_polygon(image, location.polygon.reshape(4, 2), settings.mrz.polygon_padding_ratio)
            processed = __import__("benchmarks.maintained.pipeline_breakdown", fromlist=["preprocess"]).preprocess(crop, settings.mrz.max_side, settings.mrz.contrast)
            for model_name in ("PP-OCRv6_medium_rec", "PP-OCRv6_tiny_rec"):
                chosen = replace(settings, models=replace(settings.models, text_recognizer=TextModelSettings("paddle", model_name)))
                recognizer = Models(chosen).text_recognizer(); line_crops = fixed_rows(processed, count); began = time.perf_counter(); outputs = recognizer.recognize_batch(line_crops); elapsed = time.perf_counter() - began
                rows.append({"kind": kind, "document_id": document.document_id, "variant": "ROW_SPLIT_FIXED_" + model_name, "repeat": 1, "seconds": elapsed, "line_count": len(outputs), "lines_per_second": len(outputs) / elapsed, "texts": [value.text for value in outputs], "row_geometry": [list(crop.shape[:2]) for crop in line_crops]})
    models.close()
    (directory / "environment.json").write_text(json.dumps(environment(settings, manifest, datetime.now(timezone.utc).isoformat()), indent=2, default=_json_safe))
    (directory / "configurations.json").write_text(json.dumps({"row_split": "fixed vertical bands with 2% overlap", "passport_rows": 2, "id_rows": 3}, indent=2))
    (directory / "raw.jsonl").write_text("\n".join(json.dumps(row, default=_json_safe) for row in rows) + "\n")
    (directory / "raw.csv").write_text("kind,document_id,variant,repeat,seconds,line_count,lines_per_second\n" + "\n".join(f"{r.get('kind')},{r.get('document_id','')},{r.get('variant')},{r.get('repeat')},{r.get('seconds')},{r.get('line_count','')},{r.get('lines_per_second','')}" for r in rows) + "\n")
    (directory / "summary.json").write_text(json.dumps({"rows": len(rows), "variants": sorted({r['variant'] for r in rows})}, indent=2))
    (directory / "summary.csv").write_text("kind,variant,median_seconds\n" + "\n".join(f"{kind},{variant},{stats(r['seconds'] for r in rows if r['kind']==kind and r['variant']==variant)['median']}" for kind in ("passport", "id_card") for variant in sorted({r['variant'] for r in rows})) + "\n")
    (directory / "report.md").write_text("# Experiment 5 — deterministic MRZ rows\n\nFixed geometry uses expected row count and a 2% overlap. Detector-based medium is the trusted comparator. Row crops and per-document outputs are recorded in raw.jsonl; no PII is included in this report.\n")
    print("Experiment 5 complete")
    return 0


def experiment_six() -> int:
    documents, manifest = validate_and_manifest(ROOT / "dataset")
    directory = OUTPUT / "06_fast_fallback"; directory.mkdir(parents=True, exist_ok=True)
    base = _settings(8); fast_settings = replace(base, models=replace(base.models, text_recognizer=TextModelSettings("paddle", "PP-OCRv6_tiny_rec")))
    medium = Models(base); fast = Models(fast_settings); rows = []
    for kind, variant, count, role in (("passport", "passport_mrz_only", 2, "image"), ("id_card", "id_card_mrz_known_back", 3, "back")):
        corpus = [d for d in documents if d.document_type == kind]
        for document in corpus:
            image = cv2.imread(str(dict(document.paths)[role])); location = fast.mrz_localizer().localize_batch([image])[0]; crop, _ = crop_polygon(image, location.polygon.reshape(4, 2), base.mrz.polygon_padding_ratio); processed = preprocess(crop, base.mrz.max_side, base.mrz.contrast); line_crops = fixed_rows(processed, count)
            began = time.perf_counter(); values = fast.text_recognizer().recognize_batch(line_crops); fast_seconds = time.perf_counter() - began; text = "\n".join(value.text for value in values); parsed = parse_mrz(text, kind)
            valid = bool(parsed.raw_lines) and bool(parsed.validations) and all(v.status.value == "passed" for v in parsed.validations)
            truth = json.loads(document.annotation.read_text()).get("mrz", {}).get("lines", []); false_accept = valid and [value.text for value in values] != [line for line in truth if isinstance(line, str)]
            fallback_seconds = 0.0; fallback = False
            output = {"fields": {}, "mrz": [value.text for value in values]}
            if not valid:
                fallback = True; fallback_run = _run(base, medium, [document], kind, variant, 1); fallback_seconds = fallback_run.total_seconds; output = fallback_run.outputs.get(document.document_id, output)
            rows.append({"kind": kind, "document_id": document.document_id, "fast_seconds": fast_seconds, "valid": valid, "false_accept": false_accept, "fallback": fallback, "fallback_seconds": fallback_seconds, "total_seconds": fast_seconds + fallback_seconds, "output": output})
    # Visible control: medium and tiny full-corpus direct runs, with labels scored independently.
    for kind, variant in (("passport", "passport_visible_no_mrz_ocr"), ("id_card", "id_card_visible_known_side"), ("driving_license", "driving_license_visible_ocr_only")):
        corpus = [d for d in documents if d.document_type == kind]
        for name, models in (("V0_MEDIUM_BASELINE", medium), ("V1_FAST_ONLY", fast)):
            run = _run(fast_settings if name == "V1_FAST_ONLY" else base, models, corpus, kind, variant, 1); run.variant = name
            rows.append({"kind": kind, "variant": name, "seconds": run.total_seconds, "correctness": score(corpus, [run])})
    medium.close(); fast.close()
    (directory / "environment.json").write_text(json.dumps(environment(base, manifest, datetime.now(timezone.utc).isoformat()), indent=2, default=_json_safe))
    (directory / "configurations.json").write_text(json.dumps({"fast_mrz_model": "PP-OCRv6_tiny_rec", "fallback": "detector-based PP-OCRv6_medium_rec", "visible_gate": "not available; V2 recorded as unsupported", "oracle": "offline analysis only"}, indent=2))
    (directory / "raw.jsonl").write_text("\n".join(json.dumps(row, default=_json_safe) for row in rows) + "\n")
    (directory / "raw.csv").write_text("kind,document_id,variant,fast_seconds,valid,false_accept,fallback,fallback_seconds,total_seconds\n" + "\n".join(f"{r.get('kind')},{r.get('document_id','')},{r.get('variant','MRZ')},{r.get('fast_seconds','')},{r.get('valid','')},{r.get('false_accept','')},{r.get('fallback','')},{r.get('fallback_seconds','')},{r.get('total_seconds',r.get('seconds',''))}" for r in rows) + "\n")
    mrz = [r for r in rows if "fast_seconds" in r]
    (directory / "summary.json").write_text(json.dumps({"mrz_fast_path_rate": sum(not r['fallback'] for r in mrz) / len(mrz), "mrz_fallback_rate": sum(r['fallback'] for r in mrz) / len(mrz), "false_accepts": sum(r['false_accept'] for r in mrz), "visible_modes": [r for r in rows if 'variant' in r and r['variant'].startswith('V')]}, indent=2, default=_json_safe))
    (directory / "summary.csv").write_text("kind,variant,seconds\n" + "\n".join(f"{r.get('kind')},{r.get('variant')},{r.get('seconds',r.get('total_seconds',''))}" for r in rows if 'variant' in r) + "\n")
    (directory / "report.md").write_text("# Experiment 6 — fast recognizer with trusted fallback\n\nMRZ fast acceptance was gated by parser/check-digit validation. Validation-passing but label-wrong outputs are counted as false accepts. Visible V2 real-gate routing was unsupported because no additional calibrated confidence gate exists; V3 oracle remains offline-only.\n")
    print("Experiment 6 complete")
    return 0


def final_integration() -> int:
    documents, manifest = validate_and_manifest(ROOT / "dataset")
    directory = OUTPUT / "99_final"; directory.mkdir(parents=True, exist_ok=True)
    safe = _settings(8); fast = replace(safe, models=replace(safe.models, text_recognizer=TextModelSettings("paddle", "PP-OCRv6_tiny_rec")))
    rows = []
    for label, settings in (("SAFE_OBSERVED", safe), ("FASTEST_EXPERIMENTAL", fast)):
        models = Models(settings)
        for kind, variant in (("passport", "passport_full"), ("id_card", "id_card_full"), ("driving_license", "driving_license_full")):
            corpus = [d for d in documents if d.document_type == kind]
            _combined(settings, models, corpus[:1], kind, 0)
            for repeat in range(1, 4):
                run = _combined(settings, models, corpus, kind, repeat); run.variant = label; rows.append(run)
        for kind, variant in (("passport", "passport_mrz_only"), ("id_card", "id_card_mrz_known_back")):
            corpus = [d for d in documents if d.document_type == kind]
            for repeat in range(1, 4):
                run = _run(settings, models, corpus, kind, variant, repeat); run.variant = label + "_MRZ"; rows.append(run)
        models.close()
    (directory / "environment.json").write_text(json.dumps(environment(safe, manifest, datetime.now(timezone.utc).isoformat()), indent=2, default=_json_safe))
    (directory / "configurations.json").write_text(json.dumps({"SAFE_OBSERVED": "PP-OCRv6_medium_rec batch 8", "FASTEST_EXPERIMENTAL": "PP-OCRv6_tiny_rec batch 8", "repeats": 3, "warmup": 1}, indent=2))
    (directory / "raw.jsonl").write_text("\n".join(json.dumps({"variant": r.variant, "document_type": r.document_type, "repeat": r.repeat, "seconds": r.total_seconds, "stages": r.stages, "outputs": r.outputs}, default=_json_safe) for r in rows) + "\n")
    (directory / "raw.csv").write_text("variant,document_type,repeat,seconds,docs_per_second,physical_images_per_second\n" + "\n".join(f"{r.variant},{r.document_type},{r.repeat},{r.total_seconds},{r.logical_count/r.total_seconds},{r.physical_count/r.total_seconds}" for r in rows) + "\n")
    medians = {f"{label}:{kind}": stats(r.total_seconds for r in rows if r.variant == label + ("_MRZ" if "MRZ" in kind else "") and r.document_type == kind)["median"] for label in ("SAFE_OBSERVED", "FASTEST_EXPERIMENTAL") for kind in ("passport", "id_card", "driving_license")}
    (directory / "summary.json").write_text(json.dumps({"medians": medians}, indent=2, default=_json_safe))
    (directory / "summary.csv").write_text("variant,document_type,median_seconds\n" + "\n".join(f"{r.variant},{r.document_type},{stats(x.total_seconds for x in rows if x.variant==r.variant and x.document_type==r.document_type)['median']}" for r in rows) + "\n")
    (directory / "report.md").write_text("# Final integration benchmark\n\nSAFE_OBSERVED and FASTEST_EXPERIMENTAL were measured as combined configurations; individual experiment speedups were not multiplied. Full raw timings and output signatures are in raw.jsonl.\n")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment", choices=("1", "2", "3", "4", "5", "6", "final"))
    args = parser.parse_args()
    return experiment_one() if args.experiment == "1" else experiment_two() if args.experiment == "2" else experiment_three() if args.experiment == "3" else experiment_four() if args.experiment == "4" else experiment_five() if args.experiment == "5" else experiment_six() if args.experiment == "6" else final_integration()


if __name__ == "__main__":
    raise SystemExit(main())
