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


def _optional_path(name: str) -> Path | None:
    value = os.getenv(name)
    return None if not value else _path(name, Path(value))


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
class RuntimeSettings:
    target: str = "cpu"
    cpu_threads: int = 4
    queue_limit: int = 32
    localization_batch_size: int = 4
    text_detection_batch_size: int = 8
    text_recognition_batch_size: int = 32
    text_recognition_processes: int = 1
    gpu_id: int = 0
    text_recognition_enable_hpi: bool = False
    text_recognition_use_tensorrt: bool = False
    text_recognition_precision: str = "fp32"
    mrz_recognition_batch_size: int = 16
    text_recognition_packing: str = "aspect-ratio"


@dataclass(frozen=True)
class TextModelSettings:
    backend: str
    model: str


@dataclass(frozen=True)
class LocalizationModelSettings:
    document_backend: str = "docaligner"
    mrz_backend: str = "mrzscanner"


@dataclass(frozen=True)
class MrzModelSettings:
    recognizer_backend: str = "generic-paddle"
    recognizer_model: str = "20250221"


@dataclass(frozen=True)
class ModelSettings:
    """Typed model selections; ``directory=None`` preserves cached lookup."""

    directory: Path | None = None
    text_detector: TextModelSettings = field(
        default_factory=lambda: TextModelSettings("paddle", "PP-OCRv6_medium_det")
    )
    text_recognizer: TextModelSettings = field(
        default_factory=lambda: TextModelSettings("paddle", "PP-OCRv6_medium_rec")
    )
    localization: LocalizationModelSettings = field(default_factory=LocalizationModelSettings)
    mrz: MrzModelSettings = field(default_factory=MrzModelSettings)


@dataclass(frozen=True)
class ProfileSettings:
    passport: Path = PROJECT_ROOT / "config/documents/uz_passport/profile.json"
    id_card: Path = PROJECT_ROOT / "config/documents/uz_id_card/profile.json"


@dataclass(frozen=True)
class Settings:
    preload: bool
    artifacts: ArtifactSettings
    ocr: OcrSettings
    mrz: MrzSettings
    driving_license: DrivingLicenseSettings
    batch: BatchSettings = field(default_factory=BatchSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    profiles: ProfileSettings = field(default_factory=ProfileSettings)

    def validate_startup(self) -> None:
        if self.runtime.target not in {"cpu", "gpu"}:
            raise ValueError("RUNTIME_TARGET must be 'cpu' or 'gpu'")
        if self.ocr.device not in {"cpu", "gpu"}:
            raise ValueError("OCR_DEVICE must be 'cpu' or 'gpu'")
        if self.runtime.target != self.ocr.device:
            raise ValueError("RUNTIME_TARGET and OCR_DEVICE must select the same runtime")
        if self.runtime.gpu_id < 0:
            raise ValueError("GPU_ID must be zero or greater")
        if self.runtime.target != "cpu" and self.runtime.text_recognition_processes != 1:
            raise ValueError("TEXT_RECOGNITION_PROCESSES may exceed one only on CPU")
        if self.runtime.text_recognition_precision not in {"fp32", "fp16"}:
            raise ValueError("TEXT_RECOGNITION_PRECISION must be 'fp32' or 'fp16'")
        if self.runtime.text_recognition_packing not in {"sequential", "aspect-ratio"}:
            raise ValueError("TEXT_RECOGNITION_PACKING must be 'sequential' or 'aspect-ratio'")
        if self.runtime.target == "cpu" and (
            self.runtime.text_recognition_enable_hpi
            or self.runtime.text_recognition_use_tensorrt
            or self.runtime.text_recognition_precision != "fp32"
        ):
            raise ValueError("text-recognition HPI, TensorRT, and FP16 require RUNTIME_TARGET=gpu")

        positive = {
            "CPU_THREADS": self.runtime.cpu_threads,
            "REQUEST_QUEUE_LIMIT": self.runtime.queue_limit,
            "LOCALIZATION_BATCH_SIZE": self.runtime.localization_batch_size,
            "TEXT_DETECTION_BATCH_SIZE": self.runtime.text_detection_batch_size,
            "TEXT_RECOGNITION_BATCH_SIZE": self.runtime.text_recognition_batch_size,
            "MRZ_RECOGNITION_BATCH_SIZE": self.runtime.mrz_recognition_batch_size,
            "TEXT_RECOGNITION_PROCESSES": self.runtime.text_recognition_processes,
            "BATCH_MAX_FILES": self.batch.max_files,
            "BATCH_MAX_FILE_BYTES": self.batch.max_file_bytes,
            "BATCH_MAX_ARCHIVE_UNCOMPRESSED_BYTES": self.batch.max_archive_uncompressed_bytes,
        }
        for name, value in positive.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")
        if self.batch.max_archive_uncompressed_bytes < self.batch.max_file_bytes:
            raise ValueError(
                "BATCH_MAX_ARCHIVE_UNCOMPRESSED_BYTES must be at least BATCH_MAX_FILE_BYTES"
            )

        for name, path in {
            "PASSPORT_PROFILE": self.profiles.passport,
            "ID_CARD_PROFILE": self.profiles.id_card,
        }.items():
            if not path.is_file():
                raise ValueError(f"{name} does not exist: {path}")
        if self.models.directory is not None and not self.models.directory.is_dir():
            raise ValueError(f"MODEL_DIR does not exist: {self.models.directory}")
        selections = {
            "TEXT_DETECTOR_BACKEND": self.models.text_detector.backend,
            "TEXT_DETECTOR_MODEL": self.models.text_detector.model,
            "TEXT_RECOGNIZER_BACKEND": self.models.text_recognizer.backend,
            "TEXT_RECOGNIZER_MODEL": self.models.text_recognizer.model,
            "DOCUMENT_LOCALIZER_BACKEND": self.models.localization.document_backend,
            "MRZ_LOCALIZER_BACKEND": self.models.localization.mrz_backend,
            "MRZ_RECOGNIZER_BACKEND": self.models.mrz.recognizer_backend,
            "MRZ_RECOGNIZER_MODEL": self.models.mrz.recognizer_model,
        }
        for name, value in selections.items():
            if not value.strip():
                raise ValueError(f"{name} cannot be empty")

    @classmethod
    def from_env(cls) -> "Settings":
        legacy_device = os.getenv("OCR_DEVICE")
        target = os.getenv("RUNTIME_TARGET", legacy_device or "cpu").strip().lower()
        device = (legacy_device or target).strip().lower()
        settings = cls(
            preload=_bool("PRELOAD", False),
            artifacts=ArtifactSettings(
                _bool("LOGGING", True),
                _path("LOG_DIR", PROJECT_ROOT / "logs"),
            ),
            ocr=OcrSettings(device),
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
            runtime=RuntimeSettings(
                target=target,
                cpu_threads=_positive_int("CPU_THREADS", 4),
                queue_limit=_positive_int("REQUEST_QUEUE_LIMIT", 32),
                localization_batch_size=_positive_int("LOCALIZATION_BATCH_SIZE", 4),
                text_detection_batch_size=_positive_int("TEXT_DETECTION_BATCH_SIZE", 8),
                text_recognition_batch_size=_positive_int(
                    "TEXT_RECOGNITION_BATCH_SIZE", 32
                ),
                mrz_recognition_batch_size=_positive_int(
                    "MRZ_RECOGNITION_BATCH_SIZE", 16
                ),
                text_recognition_processes=_positive_int(
                    "TEXT_RECOGNITION_PROCESSES", 1
                ),
                gpu_id=int(os.getenv("GPU_ID", "0")),
                text_recognition_enable_hpi=_bool("TEXT_RECOGNITION_ENABLE_HPI", False),
                text_recognition_use_tensorrt=_bool("TEXT_RECOGNITION_USE_TENSORRT", False),
                text_recognition_precision=os.getenv("TEXT_RECOGNITION_PRECISION", "fp32").strip().lower(),
                text_recognition_packing=os.getenv("TEXT_RECOGNITION_PACKING", "aspect-ratio").strip().lower(),
            ),
            models=ModelSettings(
                _optional_path("MODEL_DIR"),
                TextModelSettings(
                    os.getenv("TEXT_DETECTOR_BACKEND", "paddle").strip().lower(),
                    os.getenv("TEXT_DETECTOR_MODEL", "PP-OCRv6_medium_det").strip(),
                ),
                TextModelSettings(
                    os.getenv("TEXT_RECOGNIZER_BACKEND", "paddle").strip().lower(),
                    os.getenv("TEXT_RECOGNIZER_MODEL", "PP-OCRv6_medium_rec").strip(),
                ),
                LocalizationModelSettings(
                    os.getenv("DOCUMENT_LOCALIZER_BACKEND", "docaligner").strip().lower(),
                    os.getenv("MRZ_LOCALIZER_BACKEND", "mrzscanner").strip().lower(),
                ),
                MrzModelSettings(
                    os.getenv("MRZ_RECOGNIZER_BACKEND", "generic-paddle").strip().lower(),
                    os.getenv("MRZ_RECOGNIZER_MODEL", "20250221").strip(),
                ),
            ),
            profiles=ProfileSettings(
                _path(
                    "PASSPORT_PROFILE",
                    PROJECT_ROOT / "config/documents/uz_passport/profile.json",
                ),
                _path(
                    "ID_CARD_PROFILE",
                    PROJECT_ROOT / "config/documents/uz_id_card/profile.json",
                ),
            ),
        )
        settings.validate_startup()
        return settings
