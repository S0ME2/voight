"""Populate model caches during an image build, never at service startup."""

from dataclasses import replace

from app.config import Settings
from app.models import Models


def main() -> None:
    settings = Settings.from_env()
    models = Models(settings)
    models.profile_batch_runner()
    if settings.models.mrz.recognizer_backend != "mrzscanner":
        Models(
            replace(
                settings,
                models=replace(
                    settings.models,
                    mrz=replace(settings.models.mrz, recognizer_backend="mrzscanner"),
                ),
            )
        ).mrz_recognizer()


if __name__ == "__main__":
    main()
