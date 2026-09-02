"""Prepare and smoke-test every CPU model used by the alternative benchmark."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import shutil
from typing import Any

import cv2
import numpy as np

from app.config import Settings


PADDLE_DETECTORS = (
    "PP-OCRv6_medium_det",
    "PP-OCRv6_small_det",
    "PP-OCRv6_tiny_det",
    "PP-OCRv5_server_det",
    "PP-OCRv5_mobile_det",
    "PP-OCRv4_server_det",
    "PP-OCRv4_mobile_det",
)
PADDLE_RECOGNIZERS = (
    "PP-OCRv6_medium_rec",
    "PP-OCRv6_small_rec",
    "PP-OCRv6_tiny_rec",
    "PP-OCRv5_server_rec",
    "PP-OCRv5_mobile_rec",
    "latin_PP-OCRv5_mobile_rec",
    "en_PP-OCRv5_mobile_rec",
    "cyrillic_PP-OCRv5_mobile_rec",
    "eslav_PP-OCRv5_mobile_rec",
    "PP-OCRv4_server_rec",
    "PP-OCRv4_mobile_rec",
    "en_PP-OCRv4_mobile_rec",
    # The benchmark documents contain both Latin and Cyrillic text.
    "latin_PP-OCRv3_mobile_rec",
    "cyrillic_PP-OCRv3_mobile_rec",
)
DOCALIGNER_HEATMAPS = ("fastvit_sa24", "mobilenetv2_140", "fastvit_t8", "lcnet100", "lcnet050")
DOCALIGNER_POINT = "lcnet050"
SMOKE_IMAGES = {
    "paddle": Path("annotation_input/passports/passport.png"),
    "docaligner": Path("annotation_input/driving_licenses/test_license_canonical.jpg"),
    "mrzscanner": Path("annotation_input/passports/passport.png"),
}
MANIFEST_NAME = "voight-benchmark-models.json"


def _valid_paddle_dir(path: Path) -> bool:
    return all((path / name).is_file() and (path / name).stat().st_size > 0 for name in (
        "inference.yml", "inference.json", "inference.pdiparams"
    ))


def _close(model: Any) -> None:
    close = getattr(model, "close", None)
    if close is not None:
        close()
    del model
    gc.collect()


def _image(kind: str) -> np.ndarray:
    path = SMOKE_IMAGES[kind]
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"smoke image does not exist or is unreadable: {path}")
    return image


def _smoke_paddle(kind: str, name: str, directory: Path) -> None:
    from paddleocr import TextDetection, TextRecognition

    cls = TextDetection if kind == "detector" else TextRecognition
    model = cls(model_name=name, model_dir=str(directory), device="cpu")
    try:
        results = list(model.predict(input=[_image("paddle")], batch_size=1))
        if len(results) != 1:
            raise RuntimeError(f"expected one Paddle result, got {len(results)}")
    finally:
        _close(model)


def _discover_docaligner() -> tuple[list[str], list[str]]:
    from capybara import Backend
    from docaligner import DocAligner, ModelType

    heatmap_probe = DocAligner(
        model_type=ModelType.heatmap,
        model_cfg="fastvit_sa24",
        backend=Backend.cpu,
    )
    try:
        heatmaps = heatmap_probe.list_models()
    finally:
        _close(heatmap_probe)
    point_probe = DocAligner(
        model_type=ModelType.point,
        model_cfg=DOCALIGNER_POINT,
        backend=Backend.cpu,
    )
    try:
        points = point_probe.list_models()
    finally:
        _close(point_probe)
    return heatmaps, points


def _docaligner_path(model_type: str, name: str) -> Path:
    from importlib import import_module

    if model_type == "heatmap":
        module = import_module("docaligner.heatmap_reg.infer")
    else:
        module = import_module("docaligner.point_reg.infer")
    return Path(module.__file__).parent / "ckpt" / module.Inference.configs[name]["model_path"]


def _smoke_docaligner(model_type: str, name: str) -> None:
    from capybara import Backend
    from docaligner import DocAligner, ModelType

    model = DocAligner(
        model_type=ModelType.point if model_type == "point" else ModelType.heatmap,
        model_cfg=name,
        backend=Backend.cpu,
    )
    try:
        result = model(_image("docaligner"))
        if not isinstance(result, np.ndarray):
            raise RuntimeError(f"unexpected DocAligner result type: {type(result)!r}")
    finally:
        _close(model)


def _discover_mrzscanner() -> dict[str, list[str]]:
    from capybara import Backend
    from mrzscanner import MRZScanner, ModelType

    probe = MRZScanner(model_type=ModelType.detection, backend=Backend.cpu)
    try:
        models = probe.list_models()
    finally:
        _close(probe)
    return models


def _mrz_path(kind: str, name: str) -> Path:
    from importlib import import_module

    module_name = {"detection": "det", "recognition": "rec"}.get(kind, kind)
    module = import_module(f"mrzscanner.{module_name}.infer")
    return Path(module.__file__).parent / "ckpt" / module.Inference.configs[name]["model_path"]


def _smoke_mrzscanner(kind: str, name: str) -> None:
    from capybara import Backend
    from mrzscanner import MRZScanner, ModelType

    model_type = {
        "detection": ModelType.detection,
        "recognition": ModelType.recognition,
        "spotting": ModelType.spotting,
    }[kind]
    kwargs = {f"{kind}_cfg": name}
    model = MRZScanner(model_type=model_type, backend=Backend.cpu, **kwargs)
    try:
        result = model(_image("mrzscanner"))
        if not isinstance(result, dict):
            raise RuntimeError(f"unexpected MRZScanner result type: {type(result)!r}")
    finally:
        _close(model)


def _prepare_paddle(settings: Settings) -> list[dict[str, str]]:
    if settings.models.directory is None:
        raise ValueError("MODEL_DIR is required; use an explicit CPU model cache")
    from paddlex.modules.text_detection.model_list import MODELS as detector_models
    from paddlex.modules.text_recognition.model_list import MODELS as recognizer_models
    from paddlex.utils.cache import CACHE_DIR
    from paddleocr import TextDetection, TextRecognition

    missing = [name for name in PADDLE_DETECTORS if name not in detector_models]
    missing += [name for name in PADDLE_RECOGNIZERS if name not in recognizer_models]
    if missing:
        raise RuntimeError(f"installed PaddleOCR does not expose required models: {missing}")

    source_root = Path(CACHE_DIR) / "official_models"
    target_root = settings.models.directory / "official_models"
    rows = []
    for kind, names, cls in (
        ("detector", PADDLE_DETECTORS, TextDetection),
        ("recognizer", PADDLE_RECOGNIZERS, TextRecognition),
    ):
        for name in names:
            target = target_root / name
            source = source_root / name
            was_cached = _valid_paddle_dir(target)
            downloaded = False
            if not was_cached:
                if not _valid_paddle_dir(source):
                    downloader = cls(model_name=name, device="cpu")
                    _close(downloader)
                    downloaded = True
                if not _valid_paddle_dir(source):
                    raise FileNotFoundError(f"Paddle model cache is incomplete after download: {source}")
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(source, target, dirs_exist_ok=True)
            if not _valid_paddle_dir(target):
                raise FileNotFoundError(f"Paddle model cache is incomplete: {target}")
            _smoke_paddle(kind, name, target)
            rows.append({"category": f"paddle_{kind}", "model": name, "downloaded": str(downloaded), "smoke_test": "passed"})
            print(f"paddle_{kind}: {name}: {'downloaded' if downloaded else 'cached'}; smoke test passed")
    return rows


def _prepare_docaligner() -> tuple[list[dict[str, str]], dict[str, list[str]]]:
    heatmaps, points = _discover_docaligner()
    unsupported = {"heatmap": [name for name in DOCALIGNER_HEATMAPS if name not in heatmaps]}
    if DOCALIGNER_POINT not in points:
        unsupported["point"] = [DOCALIGNER_POINT]
    rows = []
    for model_type, names in (("heatmap", DOCALIGNER_HEATMAPS), ("point", (DOCALIGNER_POINT,))):
        for name in names:
            if name not in (heatmaps if model_type == "heatmap" else points):
                continue
            path = _docaligner_path(model_type, name)
            existed = path.is_file() and path.stat().st_size > 0
            _smoke_docaligner(model_type, name)
            rows.append({"category": f"docaligner_{model_type}", "model": name, "downloaded": str(not existed), "smoke_test": "passed"})
            print(f"docaligner_{model_type}: {name}: {'downloaded' if not existed else 'cached'}; smoke test passed")
    return rows, unsupported


def _prepare_mrzscanner() -> list[dict[str, str]]:
    available = _discover_mrzscanner()
    rows = []

    for kind, names in available.items():
        for name in names:
            path = _mrz_path(kind, name)
            existed = path.is_file() and path.stat().st_size > 0
            _smoke_mrzscanner(kind, name)
            rows.append({"category": f"mrzscanner_{kind}", "model": name, "downloaded": str(not existed), "smoke_test": "passed"})
            print(f"mrzscanner_{kind}: {name}: {'downloaded' if not existed else 'cached'}; smoke test passed")
    return rows


def verify_only(settings: Settings) -> list[dict[str, str]]:
    if settings.models.directory is None:
        raise ValueError("MODEL_DIR is required; use the benchmark cache directory")
    rows = []
    target_root = settings.models.directory / "official_models"
    for kind, names in (("detector", PADDLE_DETECTORS), ("recognizer", PADDLE_RECOGNIZERS)):
        for name in names:
            path = target_root / name
            if not _valid_paddle_dir(path):
                raise FileNotFoundError(f"missing Paddle model: {path}")
            _smoke_paddle(kind, name, path)
            rows.append({"category": f"paddle_{kind}", "model": name, "downloaded": "no", "smoke_test": "passed"})
    from docaligner.heatmap_reg.infer import Inference as HeatmapInference
    from docaligner.point_reg.infer import Inference as PointInference
    for model_type, configs in (("heatmap", HeatmapInference.configs), ("point", PointInference.configs)):
        names = DOCALIGNER_HEATMAPS if model_type == "heatmap" else (DOCALIGNER_POINT,)
        for name in names:
            if name not in configs:
                continue
            path = _docaligner_path(model_type, name)
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"missing DocAligner model: {path}")
            _smoke_docaligner(model_type, name)
            rows.append({"category": f"docaligner_{model_type}", "model": name, "downloaded": "no", "smoke_test": "passed"})
    from mrzscanner.det.infer import Inference as DetectionInference
    from mrzscanner.rec.infer import Inference as RecognitionInference
    from mrzscanner.spotting.infer import Inference as SpottingInference
    for kind, configs in (("detection", DetectionInference.configs), ("recognition", RecognitionInference.configs), ("spotting", SpottingInference.configs)):
        for name in configs:
            path = _mrz_path(kind, name)
            if not path.is_file() or path.stat().st_size == 0:
                raise FileNotFoundError(f"missing MRZScanner model: {path}")
            _smoke_mrzscanner(kind, name)
            rows.append({"category": f"mrzscanner_{kind}", "model": name, "downloaded": "no", "smoke_test": "passed"})
    return rows


def unsupported_docaligner() -> dict[str, list[str]]:
    from docaligner.heatmap_reg.infer import Inference as HeatmapInference
    from docaligner.point_reg.infer import Inference as PointInference

    unsupported = {"heatmap": [name for name in DOCALIGNER_HEATMAPS if name not in HeatmapInference.configs]}
    if DOCALIGNER_POINT not in PointInference.configs:
        unsupported["point"] = [DOCALIGNER_POINT]
    return unsupported


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true", help="verify caches without downloading")
    args = parser.parse_args()
    settings = Settings.from_env()
    if args.verify_only:
        rows = verify_only(settings)
        unsupported = unsupported_docaligner()
    else:
        rows = _prepare_paddle(settings)
        docaligner_rows, unsupported = _prepare_docaligner()
        rows += docaligner_rows + _prepare_mrzscanner()
    manifest = {"models": rows, "unsupported": unsupported}
    if settings.models.directory is not None:
        (settings.models.directory / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
