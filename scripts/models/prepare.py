"""Populate and verify model caches during an image build."""

from dataclasses import replace
import json
from pathlib import Path
import shutil

from app.config import Settings
from app.models import Models


def required_paddle_models(settings: Settings) -> tuple[Path, ...]:
    if settings.models.directory is None:
        raise ValueError("MODEL_DIR is required while preparing image models")
    root = settings.models.directory / "official_models"
    return (
        root / settings.models.text_detector.model,
        root / settings.models.text_recognizer.model,
    )


def verify_models(settings: Settings) -> tuple[Path, ...]:
    paths = required_paddle_models(settings)
    missing = [path for path in paths if not path.is_dir() or not any(path.iterdir())]
    if missing:
        raise FileNotFoundError("missing prepared model directories: " + ", ".join(map(str, missing)))
    return paths


def main() -> None:
    settings = Settings.from_env()
    try:
        target = settings.models.directory
        if target is None:
            raise ValueError("MODEL_DIR is required while preparing image models")
        download_settings = replace(
            settings,
            models=replace(settings.models, directory=None),
        )
        models = Models(download_settings)
        models.profile_batch_runner()
        if download_settings.models.mrz.recognizer_backend != "mrzscanner":
            Models(
                replace(
                    download_settings,
                    models=replace(
                        download_settings.models,
                        mrz=replace(download_settings.models.mrz, recognizer_backend="mrzscanner"),
                    ),
                )
            ).mrz_recognizer()
        source = Path.home() / ".paddlex" / "official_models"
        for path in required_paddle_models(settings):
            shutil.copytree(source / path.name, path, dirs_exist_ok=True)
        paths = verify_models(settings)
        manifest = {
            "paddle_models": [str(path.relative_to(settings.models.directory)) for path in paths],
            "document_localizer": settings.driving_license.aligner_model,
            "mrz_detector": settings.mrz.scanner_config,
            "mrz_recognizer": settings.models.mrz.recognizer_model,
        }
        (settings.models.directory / "voight-models.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
    except Exception as error:
        root = settings.models.directory or Path("<unset>")
        raise SystemExit(
            "Failed to prepare Voight OCR models.\n"
            f"Model source: pinned PaddleOCR/DocSaid packages\nExpected container path: {root}\n"
            "Run: make models-cpu\n"
            f"Cause: {error}"
        ) from error


if __name__ == "__main__":
    main()
