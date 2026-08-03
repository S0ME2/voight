import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    return default if value is None else value.strip().lower() in {"1", "true", "yes", "on"}


def _positive_int(name: str, default: int) -> int:
    value = int(os.getenv(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be greater than zero")
    return value


def _path(name: str, default: Path) -> Path:
    value = Path(os.getenv(name, str(default))).expanduser()
    return value.resolve() if value.is_absolute() else (PROJECT_ROOT / value).resolve()


@dataclass(frozen=True)
class ArtifactSettings:
    enabled: bool
    directory: Path


@dataclass(frozen=True)
class OcrSettings:
    device: str


@dataclass(frozen=True)
class MrzSettings:
    scanner_config: str
    max_side: int
    contrast: float
    polygon_padding_ratio: float


@dataclass(frozen=True)
class DrivingLicenseSettings:
    data_crop: Path
    field_rois: Path
    canonical_width: int
    canonical_height: int
    aligner_padding: int
    min_overlap_ratio: float
    aligner_model: str


@dataclass(frozen=True)
class BatchSettings:
    max_files: int = 20
    max_file_bytes: int = 50 * 1024 * 1024
    max_archive_uncompressed_bytes: int = 500 * 1024 * 1024


@dataclass(frozen=True)
class Settings:
    preload: bool
    artifacts: ArtifactSettings
    ocr: OcrSettings
    mrz: MrzSettings
    driving_license: DrivingLicenseSettings
    batch: BatchSettings = field(default_factory=BatchSettings)

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            preload=_bool("PRELOAD", False),
            artifacts=ArtifactSettings(
                _bool("LOGGING", True),
                _path("LOG_DIR", PROJECT_ROOT / "logs"),
            ),
            ocr=OcrSettings(os.getenv("OCR_DEVICE", "cpu")),
            mrz=MrzSettings(
                os.getenv("MRZSCANNER_DETECTION_CFG", "20250222"),
                int(os.getenv("OCR_MAX_SIDE", "3000")),
                float(os.getenv("OCR_CONTRAST", "1.25")),
                float(os.getenv("MRZ_POLYGON_PADDING_RATIO", "0.03")),
            ),
            driving_license=DrivingLicenseSettings(
                _path(
                    "DRIVING_LICENSE_DATA_CROP",
                    PROJECT_ROOT / "config/driving_license/data_crop.json",
                ),
                _path(
                    "DRIVING_LICENSE_FIELD_ROIS",
                    PROJECT_ROOT / "config/driving_license/field_rois_crop.json",
                ),
                int(os.getenv("DRIVING_LICENSE_CANONICAL_WIDTH", "1000")),
                int(os.getenv("DRIVING_LICENSE_CANONICAL_HEIGHT", "630")),
                int(os.getenv("DOCALIGNER_PADDING", "100")),
                float(os.getenv("DRIVING_LICENSE_MIN_OVERLAP_RATIO", "0.30")),
                os.getenv("DOCALIGNER_MODEL", "fastvit_sa24"),
            ),
            batch=BatchSettings(
                max_files=_positive_int("BATCH_MAX_FILES", 20),
                max_file_bytes=_positive_int(
                    "BATCH_MAX_FILE_BYTES", 50 * 1024 * 1024
                ),
                max_archive_uncompressed_bytes=_positive_int(
                    "BATCH_MAX_ARCHIVE_UNCOMPRESSED_BYTES", 500 * 1024 * 1024
                ),
            ),
        )
