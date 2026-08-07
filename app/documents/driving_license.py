import time
from typing import Any

import numpy as np

from app.artifacts import ArtifactWriter
from app.config import DrivingLicenseSettings
from app.documents.driving_license_fields import parse_fields, validation_warnings
from app.ocr import recognize
from app.pipeline import extract_profile, load_region_profile


def extract(
    image: np.ndarray,
    aligner: Any,
    ocr: Any,
    artifacts: ArtifactWriter,
    settings: DrivingLicenseSettings,
    *,
    started_total: float | None = None,
    initial_timings: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    started_total = started_total if started_total is not None else time.perf_counter()
    profile = load_region_profile(settings.data_crop, settings.field_rois)
    return extract_profile(
        image,
        profile,
        lambda padded: aligner(img=padded, do_center_crop=False),
        lambda data_crop: recognize(data_crop, ocr, artifacts),
        parse_fields,
        validation_warnings,
        artifacts,
        canonical_width=settings.canonical_width,
        canonical_height=settings.canonical_height,
        padding=settings.aligner_padding,
        min_overlap=settings.min_overlap_ratio,
        started_total=started_total,
        initial_timings=initial_timings,
    )
