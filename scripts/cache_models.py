"""Populate the Paddle caches during an image build, never at service startup."""

from app.config import Settings
from app.models import Models


def main() -> None:
    models = Models(Settings.from_env())
    models.ocr()
    models.mrz_scanner()
    models.document_aligner()
    models.profile_batch_runner()


if __name__ == "__main__":
    main()
