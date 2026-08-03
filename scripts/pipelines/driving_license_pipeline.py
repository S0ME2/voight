from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# =========================================================
# CONFIGURATION
# =========================================================


@dataclass(frozen=True)
class PipelineConfig:
    data_crop_path: str = "config/driving_license/data_crop.json"
    field_rois_path: str = "config/driving_license/field_rois_crop.json"

    output_root: str = "outputs/pipelines/driving_license"

    canonical_width: int = 1000
    canonical_height: int = 630

    # Padding helps DocAligner when the document
    # is close to the photograph boundaries.
    docaligner_padding: int = 100

    # Minimum fraction of an OCR box that must overlap
    # a field ROI before it can be assigned to that field.
    min_overlap_ratio: float = 0.30

    docaligner_model: str = "fastvit_sa24"


# =========================================================
# FIELD PREFIXES
# =========================================================
#
# IMPORTANT:
#
# These prefixes are NOT used to determine which field
# a text belongs to.
#
# Position / ROI overlap is the primary signal.
#
# Prefixes are only removed AFTER assignment.
# Therefore OCR can fail to recognize "4a.", "5.", etc.
# without breaking field assignment.
# =========================================================


FIELD_PREFIXES: dict[str, list[str]] = {
    "surname": [
        r"^\s*1\s*[.\-:,]?\s*",
    ],
    "given_names": [
        r"^\s*2\s*[.\-:,]?\s*",
    ],
    "birth_place_and_date": [
        r"^\s*3\s*[.\-:,]?\s*",
    ],
    "issue_date": [
        r"^\s*4\s*[aA]\s*[.\-:,]?\s*",
    ],
    "expiry_date": [
        r"^\s*4\s*[bB]\s*[.\-:,]?\s*",
    ],
    "issued_place": [
        r"^\s*4\s*[cC]\s*[.\-:,]?\s*",
    ],
    "personal_id": [
        r"^\s*4\s*[dD]\s*[.\-:,]?\s*",
    ],
    "license_number": [
        r"^\s*5\s*[.\-:,]?\s*",
    ],
    "address": [
        r"^\s*8\s*[.\-:,]?\s*",
    ],
    "categories": [
        r"^\s*9\s*[.\-:,]?\s*",
    ],
}


# Handles:
#
# 19.10.2005
# 19. 10. 2005
# 19/10/2005
# 19-10-2005
#
DATE_PATTERN = re.compile(
    r"(?<!\d)"
    r"(\d{1,2})"
    r"\s*[.\-/]\s*"
    r"(\d{1,2})"
    r"\s*[.\-/]\s*"
    r"(\d{4})"
    r"(?!\d)"
)


# =========================================================
# GENERAL JSON HELPERS
# =========================================================


def load_json(path: str | Path) -> dict:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")

    with path.open(
        "r",
        encoding="utf-8",
    ) as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in: {path}")

    return data


def save_json(
    path: str | Path,
    data: Any,
) -> None:
    path = Path(path)

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=4,
        )


# =========================================================
# DOCALIGNER / PERSPECTIVE HELPERS
# =========================================================


def order_points(
    points: np.ndarray,
) -> np.ndarray:
    """
    Order document corners as:

        top-left
        top-right
        bottom-right
        bottom-left
    """

    points = np.asarray(
        points,
        dtype=np.float32,
    ).reshape(4, 2)

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32,
    )

    point_sum = points.sum(axis=1)

    point_diff = np.diff(
        points,
        axis=1,
    ).reshape(-1)

    # Smallest x + y
    ordered[0] = points[np.argmin(point_sum)]

    # Largest x + y
    ordered[2] = points[np.argmax(point_sum)]

    # Smallest y - x
    ordered[1] = points[np.argmin(point_diff)]

    # Largest y - x
    ordered[3] = points[np.argmax(point_diff)]

    return ordered


def draw_document_detection(
    image: np.ndarray,
    corners: np.ndarray,
) -> np.ndarray:
    """
    Draw detected document polygon and corner labels.
    """

    output = image.copy()

    ordered = order_points(corners)

    polygon = ordered.astype(np.int32)

    cv2.polylines(
        output,
        [polygon],
        isClosed=True,
        color=(0, 255, 0),
        thickness=4,
    )

    labels = [
        "TL",
        "TR",
        "BR",
        "BL",
    ]

    for label, point in zip(
        labels,
        ordered,
    ):
        x, y = map(
            int,
            point,
        )

        cv2.circle(
            output,
            (x, y),
            8,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            output,
            label,
            (
                x + 10,
                max(20, y - 10),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return output


def warp_to_canonical(
    image: np.ndarray,
    corners: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """
    Perspective-warp document to fixed canonical dimensions.
    """

    source = order_points(corners)

    destination = np.array(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(
        source,
        destination,
    )

    canonical = cv2.warpPerspective(
        image,
        transform,
        (
            width,
            height,
        ),
    )

    return canonical


# =========================================================
# ROI HELPERS
# =========================================================


def validate_normalized_roi(
    roi: dict,
    name: str,
) -> None:
    required_keys = {
        "x1",
        "y1",
        "x2",
        "y2",
    }

    if not required_keys.issubset(roi.keys()):
        raise ValueError(f"ROI '{name}' is missing coordinates.")

    x1 = float(roi["x1"])
    y1 = float(roi["y1"])
    x2 = float(roi["x2"])
    y2 = float(roi["y2"])

    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError(f"Invalid normalized ROI '{name}': {roi}")


def normalized_roi_to_pixels(
    roi: dict,
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """
    Convert normalized ROI coordinates into pixels.
    """

    x1 = int(round(float(roi["x1"]) * image_width))

    y1 = int(round(float(roi["y1"]) * image_height))

    x2 = int(round(float(roi["x2"]) * image_width))

    y2 = int(round(float(roi["y2"]) * image_height))

    # Defensive clamping.
    x1 = max(
        0,
        min(
            image_width - 1,
            x1,
        ),
    )

    y1 = max(
        0,
        min(
            image_height - 1,
            y1,
        ),
    )

    x2 = max(
        x1 + 1,
        min(
            image_width,
            x2,
        ),
    )

    y2 = max(
        y1 + 1,
        min(
            image_height,
            y2,
        ),
    )

    return (
        x1,
        y1,
        x2,
        y2,
    )


def crop_normalized_roi(
    image: np.ndarray,
    roi: dict,
) -> np.ndarray:
    """
    Crop image using normalized ROI coordinates.
    """

    image_height, image_width = image.shape[:2]

    x1, y1, x2, y2 = normalized_roi_to_pixels(
        roi,
        image_width,
        image_height,
    )

    crop = image[
        y1:y2,
        x1:x2,
    ].copy()

    if crop.size == 0:
        raise RuntimeError("Data crop produced an empty image.")

    return crop


# =========================================================
# OCR TOKEN HELPERS
# =========================================================


def create_ocr_tokens(
    result,
) -> list[dict]:
    """
    Convert PaddleOCR result into a convenient structure.
    """

    texts = result["rec_texts"]
    scores = result["rec_scores"]
    boxes = result["rec_boxes"]

    tokens = []

    for index, (
        text,
        score,
        box,
    ) in enumerate(
        zip(
            texts,
            scores,
            boxes,
        )
    ):
        x1, y1, x2, y2 = map(
            float,
            box,
        )

        tokens.append(
            {
                "index": index,
                "text": str(text).strip(),
                "score": float(score),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": (x1 + x2) / 2,
                "center_y": (y1 + y2) / 2,
            }
        )

    return tokens


def box_area(
    box: tuple[
        float,
        float,
        float,
        float,
    ],
) -> float:
    x1, y1, x2, y2 = box

    return max(
        0.0,
        x2 - x1,
    ) * max(
        0.0,
        y2 - y1,
    )


def intersection_area(
    box_a: tuple[
        float,
        float,
        float,
        float,
    ],
    box_b: tuple[
        float,
        float,
        float,
        float,
    ],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(
        ax1,
        bx1,
    )

    y1 = max(
        ay1,
        by1,
    )

    x2 = min(
        ax2,
        bx2,
    )

    y2 = min(
        ay2,
        by2,
    )

    if x2 <= x1 or y2 <= y1:
        return 0.0

    return (x2 - x1) * (y2 - y1)


def token_roi_overlap_ratio(
    token_box: tuple[
        float,
        float,
        float,
        float,
    ],
    roi_box: tuple[
        float,
        float,
        float,
        float,
    ],
) -> float:
    """
    Percentage of OCR token area inside the ROI.

    We intentionally divide by token area rather
    than ROI area because ROIs are generous zones.
    """

    token_area = box_area(token_box)

    if token_area <= 0:
        return 0.0

    overlap = intersection_area(
        token_box,
        roi_box,
    )

    return overlap / token_area


def assign_tokens_to_fields(
    tokens: list[dict],
    rois: dict,
    image_width: int,
    image_height: int,
    min_overlap_ratio: float,
) -> tuple[
    dict[str, list[dict]],
    list[dict],
]:
    """
    Assign each OCR token to exactly one field:

    the field ROI with the greatest overlap.

    This prevents duplicates when generous ROIs
    slightly overlap each other.
    """

    pixel_rois = {}

    for field_name, roi in rois.items():
        pixel_rois[field_name] = normalized_roi_to_pixels(
            roi,
            image_width,
            image_height,
        )

    assignments = {field_name: [] for field_name in rois}

    unassigned = []

    for token in tokens:
        token_box = (
            token["x1"],
            token["y1"],
            token["x2"],
            token["y2"],
        )

        best_field = None
        best_overlap = 0.0

        for (
            field_name,
            roi_box,
        ) in pixel_rois.items():
            overlap = token_roi_overlap_ratio(
                token_box,
                roi_box,
            )

            if overlap > best_overlap:
                best_overlap = overlap
                best_field = field_name

        if best_field is not None and best_overlap >= min_overlap_ratio:
            assignments[best_field].append(
                {
                    **token,
                    "overlap_ratio": float(best_overlap),
                }
            )

        else:
            unassigned.append(token)

    return (
        assignments,
        unassigned,
    )


# =========================================================
# TEXT MERGING / CLEANUP
# =========================================================


def merge_tokens(
    tokens: list[dict],
) -> str:
    """
    Merge OCR chunks in reading order.

    Example:

        2. HUSNIDDIN
        MIRZOHID O'G'LI

    becomes:

        2. HUSNIDDIN MIRZOHID O'G'LI
    """

    if not tokens:
        return ""

    sorted_tokens = sorted(
        tokens,
        key=lambda token: (
            token["center_y"],
            token["x1"],
        ),
    )

    return " ".join(token["text"] for token in sorted_tokens if token["text"])


def remove_field_prefix(
    field_name: str,
    value: str,
) -> str:
    """
    Remove known field labels if OCR recognized them.

    Field assignment itself does NOT depend on this.
    """

    cleaned = value.strip()

    for pattern in FIELD_PREFIXES.get(
        field_name,
        [],
    ):
        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            count=1,
            flags=re.IGNORECASE,
        )

    return cleaned.strip()


def normalize_spaces(
    value: str,
) -> str:
    return " ".join(value.split())


def normalize_text(
    value: str,
) -> str:
    """
    Normalize whitespace and punctuation spacing.
    """

    value = normalize_spaces(value)

    value = re.sub(
        r"\s*,\s*",
        ", ",
        value,
    )

    return value.strip()


# =========================================================
# FIELD PARSING
# =========================================================


def extract_date(
    value: str,
) -> str | None:
    """
    Find and validate a date.
    """

    match = DATE_PATTERN.search(value)

    if not match:
        return None

    day, month, year = map(
        int,
        match.groups(),
    )

    try:
        datetime(
            year,
            month,
            day,
        )
    except ValueError:
        return None

    return f"{day:02d}.{month:02d}.{year:04d}"


def process_simple_text(
    field_name: str,
    value: str,
) -> str | None:
    value = remove_field_prefix(
        field_name,
        value,
    )

    value = normalize_text(value)

    return value or None


def process_date_field(
    value: str,
) -> str | None:
    """
    Date extraction does not depend on the 4a/4b label.

    Even if OCR reads the label incorrectly, the
    date can still be found from the ROI contents.
    """

    return extract_date(value)


def process_birth_place_and_date(
    value: str,
) -> tuple[
    str | None,
    str | None,
]:
    value = remove_field_prefix(
        "birth_place_and_date",
        value,
    )

    match = DATE_PATTERN.search(value)

    if not match:
        birth_place = normalize_text(value)

        return (
            birth_place or None,
            None,
        )

    birth_date = extract_date(match.group(0))

    birth_place = value[: match.start()]

    birth_place = normalize_text(birth_place)

    return (
        birth_place or None,
        birth_date,
    )


def process_personal_id(
    value: str,
) -> str | None:
    """
    Extract a numeric personal ID.

    First prefer a long contiguous number.
    This avoids accidentally including the '4'
    from a failed '4d.' prefix removal.
    """

    value = remove_field_prefix(
        "personal_id",
        value,
    )

    # Common OCR corrections in numeric-only field.
    numeric_value = (
        value.upper()
        .replace(
            "O",
            "0",
        )
        .replace(
            "Q",
            "0",
        )
        .replace(
            "I",
            "1",
        )
        .replace(
            "L",
            "1",
        )
    )

    numeric_runs = re.findall(
        r"\d{8,}",
        numeric_value,
    )

    if numeric_runs:
        return max(
            numeric_runs,
            key=len,
        )

    # Fallback for OCR that inserted spaces.
    digits = re.sub(
        r"\D",
        "",
        numeric_value,
    )

    return digits or None


def process_license_number(
    value: str,
) -> str | None:
    value = remove_field_prefix(
        "license_number",
        value,
    )

    value = value.upper()

    # Prefer something like:
    # AG2742395
    matches = re.findall(
        r"[A-Z]{1,4}\d{4,}",
        value,
    )

    if matches:
        return max(
            matches,
            key=len,
        )

    cleaned = re.sub(
        r"[^A-Z0-9]",
        "",
        value,
    )

    return cleaned or None


def process_categories(
    value: str,
) -> str | None:
    value = remove_field_prefix(
        "categories",
        value,
    )

    value = normalize_text(value.upper())

    return value or None


def process_serial_number(
    value: str,
) -> str | None:
    value = value.upper()

    matches = re.findall(
        r"[A-Z]{1,4}\d{4,}",
        value,
    )

    if matches:
        return max(
            matches,
            key=len,
        )

    cleaned = re.sub(
        r"[^A-Z0-9]",
        "",
        value,
    )

    return cleaned or None


def build_extracted_data(
    assignments: dict[
        str,
        list[dict],
    ],
) -> tuple[
    dict,
    dict,
]:
    """
    Merge raw OCR tokens and create final structured JSON.
    """

    raw_fields = {
        field_name: merge_tokens(tokens)
        for (
            field_name,
            tokens,
        ) in assignments.items()
    }

    (
        birth_place,
        birth_date,
    ) = process_birth_place_and_date(
        raw_fields.get(
            "birth_place_and_date",
            "",
        )
    )

    extracted = {
        "surname": process_simple_text(
            "surname",
            raw_fields.get(
                "surname",
                "",
            ),
        ),
        "given_names": process_simple_text(
            "given_names",
            raw_fields.get(
                "given_names",
                "",
            ),
        ),
        "birth_place": birth_place,
        "birth_date": birth_date,
        "issue_date": process_date_field(
            raw_fields.get(
                "issue_date",
                "",
            )
        ),
        "expiry_date": process_date_field(
            raw_fields.get(
                "expiry_date",
                "",
            )
        ),
        "issued_place": process_simple_text(
            "issued_place",
            raw_fields.get(
                "issued_place",
                "",
            ),
        ),
        "personal_id": process_personal_id(
            raw_fields.get(
                "personal_id",
                "",
            )
        ),
        "license_number": (
            process_license_number(
                raw_fields.get(
                    "license_number",
                    "",
                )
            )
        ),
        "address": process_simple_text(
            "address",
            raw_fields.get(
                "address",
                "",
            ),
        ),
        "categories": process_categories(
            raw_fields.get(
                "categories",
                "",
            )
        ),
        "serial_number": (
            process_serial_number(
                raw_fields.get(
                    "serial_number",
                    "",
                )
            )
        ),
    }

    return (
        extracted,
        raw_fields,
    )


# =========================================================
# VALIDATION
# =========================================================


def validate_extracted_data(
    data: dict,
) -> list[str]:
    """
    Non-destructive validation.

    We do NOT remove potentially useful OCR values.
    We only generate warnings.
    """

    warnings = []

    required_fields = [
        "surname",
        "given_names",
        "birth_date",
        "issue_date",
        "expiry_date",
        "personal_id",
        "license_number",
    ]

    for field_name in required_fields:
        if not data.get(field_name):
            warnings.append(f"Missing required field: {field_name}")

    personal_id = data.get("personal_id")

    if personal_id and len(personal_id) != 14:
        warnings.append(f"personal_id has unexpected length: {len(personal_id)}")

    return warnings


# =========================================================
# VISUAL DEBUGGING
# =========================================================


def save_field_assignment_annotation(
    image: np.ndarray,
    rois: dict,
    assignments: dict[
        str,
        list[dict],
    ],
    output_path: str | Path,
) -> None:
    annotated = image.copy()

    image_height, image_width = image.shape[:2]

    for field_name, roi in rois.items():
        x1, y1, x2, y2 = normalized_roi_to_pixels(
            roi,
            image_width,
            image_height,
        )

        # Green = field ROI
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            annotated,
            field_name,
            (
                x1,
                max(
                    15,
                    y1 - 4,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        # Blue = OCR token assigned to this ROI.
        for token in assignments.get(
            field_name,
            [],
        ):
            tx1 = int(token["x1"])
            ty1 = int(token["y1"])
            tx2 = int(token["x2"])
            ty2 = int(token["y2"])

            cv2.rectangle(
                annotated,
                (tx1, ty1),
                (tx2, ty2),
                (255, 0, 0),
                1,
            )

    if not cv2.imwrite(
        str(output_path),
        annotated,
    ):
        raise RuntimeError(f"Could not save annotation: {output_path}")


# =========================================================
# MAIN PIPELINE CLASS
# =========================================================


class DrivingLicensePipeline:
    """
    End-to-end driving licence extraction pipeline.

    Models are initialized ONCE and can then be reused
    across multiple images.
    """

    def __init__(
        self,
        config: PipelineConfig,
    ):
        self.config = config

        # -----------------------------------------
        # Load crop definition
        # -----------------------------------------

        data_crop_json = load_json(config.data_crop_path)

        if "data_crop" not in data_crop_json:
            raise ValueError("data_crop.json must contain a 'data_crop' object.")

        self.data_crop_roi = data_crop_json["data_crop"]

        validate_normalized_roi(
            self.data_crop_roi,
            "data_crop",
        )

        # -----------------------------------------
        # Load field ROIs
        # -----------------------------------------

        self.field_rois = load_json(config.field_rois_path)

        if not self.field_rois:
            raise ValueError("No field ROIs found.")

        for (
            field_name,
            roi,
        ) in self.field_rois.items():
            validate_normalized_roi(
                roi,
                field_name,
            )

        # -----------------------------------------
        # Initialize models ONCE
        # -----------------------------------------

        print()
        print("=" * 70)
        print("INITIALIZING MODELS")
        print("=" * 70)

        print("Loading DocAligner...")

        from docaligner import DocAligner
        from paddleocr import PaddleOCR

        self.docaligner = DocAligner(model_cfg=(config.docaligner_model))

        print("Loading PaddleOCR...")

        self.ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

        print("Models ready.")

    # =====================================================
    # STAGE 1 — DOCUMENT DETECTION
    # =====================================================

    def detect_document(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        padding = self.config.docaligner_padding

        padded = cv2.copyMakeBorder(
            image,
            padding,
            padding,
            padding,
            padding,
            borderType=(cv2.BORDER_CONSTANT),
            value=(
                0,
                0,
                0,
            ),
        )

        polygon = self.docaligner(
            img=padded,
            do_center_crop=False,
        )

        if polygon is None:
            raise RuntimeError("DocAligner returned no polygon.")

        polygon = np.asarray(
            polygon,
            dtype=np.float32,
        )

        if polygon.size != 8:
            raise RuntimeError(f"Unexpected DocAligner output shape: {polygon.shape}")

        polygon = polygon.reshape(
            4,
            2,
        )

        # Convert padded coordinates back
        # into original-image coordinates.
        polygon[:, 0] -= padding
        polygon[:, 1] -= padding

        return polygon

    # =====================================================
    # STAGE 2 — CANONICALIZATION
    # =====================================================

    def canonicalize(
        self,
        image: np.ndarray,
        corners: np.ndarray,
    ) -> np.ndarray:
        return warp_to_canonical(
            image=image,
            corners=corners,
            width=(self.config.canonical_width),
            height=(self.config.canonical_height),
        )

    # =====================================================
    # STAGE 3 — DATA CROP
    # =====================================================

    def crop_data_region(
        self,
        canonical: np.ndarray,
    ) -> np.ndarray:
        return crop_normalized_roi(
            canonical,
            self.data_crop_roi,
        )

    # =====================================================
    # STAGE 4 — OCR
    # =====================================================

    def run_ocr(
        self,
        image_path: str | Path,
    ):
        results = list(self.ocr.predict(str(image_path)))

        if not results:
            raise RuntimeError("PaddleOCR returned no results.")

        return results[0]

    # =====================================================
    # FULL PIPELINE
    # =====================================================

    def process(
        self,
        image_path: str | Path,
        output_dir: str | Path,
    ) -> dict:
        image_path = Path(image_path)

        output_dir = Path(output_dir)

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        image = cv2.imread(str(image_path))

        if image is None:
            raise FileNotFoundError(f"Could not load image: {image_path}")

        timings = {}

        total_start = time.perf_counter()

        # =================================================
        # 1. DOCALIGNER
        # =================================================

        start = time.perf_counter()

        corners = self.detect_document(image)

        timings["document_detection_seconds"] = time.perf_counter() - start

        detection_debug = draw_document_detection(
            image,
            corners,
        )

        detection_path = output_dir / "01_document_detection.jpg"

        cv2.imwrite(
            str(detection_path),
            detection_debug,
        )

        # =================================================
        # 2. CANONICAL LICENCE
        # =================================================

        start = time.perf_counter()

        canonical = self.canonicalize(
            image,
            corners,
        )

        timings["canonicalization_seconds"] = time.perf_counter() - start

        canonical_path = output_dir / "02_canonical_license.jpg"

        cv2.imwrite(
            str(canonical_path),
            canonical,
        )

        # =================================================
        # 3. USEFUL DATA CROP
        # =================================================

        start = time.perf_counter()

        data_crop = self.crop_data_region(canonical)

        timings["data_crop_seconds"] = time.perf_counter() - start

        data_crop_path = output_dir / "03_data_crop.jpg"

        cv2.imwrite(
            str(data_crop_path),
            data_crop,
        )

        crop_height, crop_width = data_crop.shape[:2]

        # =================================================
        # 4. PADDLE OCR
        # =================================================

        start = time.perf_counter()

        result = self.run_ocr(data_crop_path)

        timings["ocr_seconds"] = time.perf_counter() - start

        paddle_annotation_path = output_dir / "04_paddle_ocr_annotated.jpg"

        result.save_to_img(str(paddle_annotation_path))

        # =================================================
        # 5. CREATE OCR TOKENS
        # =================================================

        tokens = create_ocr_tokens(result)

        raw_ocr_path = output_dir / "raw_ocr.json"

        save_json(
            raw_ocr_path,
            tokens,
        )

        # =================================================
        # 6. FIELD ASSIGNMENT
        # =================================================

        start = time.perf_counter()

        (
            assignments,
            unassigned,
        ) = assign_tokens_to_fields(
            tokens=tokens,
            rois=self.field_rois,
            image_width=crop_width,
            image_height=crop_height,
            min_overlap_ratio=(self.config.min_overlap_ratio),
        )

        timings["field_assignment_seconds"] = time.perf_counter() - start

        field_annotation_path = output_dir / "05_field_assignment_annotated.jpg"

        save_field_assignment_annotation(
            image=data_crop,
            rois=self.field_rois,
            assignments=assignments,
            output_path=(field_annotation_path),
        )

        # =================================================
        # 7. FIELD PARSING
        # =================================================

        start = time.perf_counter()

        (
            extracted,
            raw_fields,
        ) = build_extracted_data(assignments)

        timings["field_parsing_seconds"] = time.perf_counter() - start

        save_json(
            output_dir / "raw_fields.json",
            raw_fields,
        )

        save_json(
            output_dir / "extracted.json",
            extracted,
        )

        # =================================================
        # 8. VALIDATION
        # =================================================

        validation_warnings = validate_extracted_data(extracted)

        # =================================================
        # 9. REPORT
        # =================================================

        timings["total_seconds"] = time.perf_counter() - total_start

        ordered_corners = order_points(corners)

        report = {
            "input_image": str(image_path),
            "canonical_size": {
                "width": (self.config.canonical_width),
                "height": (self.config.canonical_height),
            },
            "data_crop_size": {
                "width": crop_width,
                "height": crop_height,
            },
            "detected_corners": {
                label: [
                    float(point[0]),
                    float(point[1]),
                ]
                for label, point in zip(
                    [
                        "top_left",
                        "top_right",
                        "bottom_right",
                        "bottom_left",
                    ],
                    ordered_corners,
                )
            },
            "ocr_token_count": len(tokens),
            "unassigned_token_count": len(unassigned),
            "unassigned_tokens": [
                {
                    "text": token["text"],
                    "score": token["score"],
                }
                for token in unassigned
            ],
            "validation_warnings": (validation_warnings),
            "timings": timings,
        }

        save_json(
            output_dir / "pipeline_report.json",
            report,
        )

        # =================================================
        # CONSOLE OUTPUT
        # =================================================

        print()
        print("=" * 70)
        print(f"PROCESSED: {image_path.name}")
        print("=" * 70)

        print(
            json.dumps(
                extracted,
                ensure_ascii=False,
                indent=4,
            )
        )

        if validation_warnings:
            print()
            print("Validation warnings:")

            for warning in validation_warnings:
                print(f"  - {warning}")

        print()
        print(f"Total time: {timings['total_seconds']:.2f}s")

        print(f"OCR time:   {timings['ocr_seconds']:.2f}s")

        print(f"Output:     {output_dir}")

        return extracted


# =========================================================
# CLI
# =========================================================


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=("End-to-end Uzbek driving licence OCR pipeline.")
    )

    parser.add_argument(
        "images",
        nargs="+",
        help=("One or more driving licence photographs."),
    )

    parser.add_argument(
        "--data-crop",
        default="config/driving_license/data_crop.json",
        help=("Path to data_crop.json"),
    )

    parser.add_argument(
        "--field-rois",
        default=("config/driving_license/field_rois_crop.json"),
        help=("Path to field_rois_crop.json"),
    )

    parser.add_argument(
        "--output-root",
        default="outputs/pipelines/driving_license",
        help=("Directory where results will be saved."),
    )

    parser.add_argument(
        "--canonical-width",
        type=int,
        default=1000,
        help=("Canonical driving-licence width in pixels."),
    )

    parser.add_argument(
        "--canonical-height",
        type=int,
        default=630,
        help=("Canonical driving-licence height in pixels."),
    )

    parser.add_argument(
        "--model",
        default="fastvit_sa24",
        help=("DocAligner model configuration name."),
    )

    parser.add_argument(
        "--padding",
        type=int,
        default=100,
        help=("Padding added before DocAligner inference."),
    )

    parser.add_argument(
        "--min-overlap",
        type=float,
        default=0.30,
        help=("Minimum OCR-token/ROI overlap ratio."),
    )

    return parser.parse_args()


# =========================================================
# ENTRY POINT
# =========================================================


def main():
    args = parse_arguments()

    config = PipelineConfig(
        data_crop_path=(args.data_crop),
        field_rois_path=(args.field_rois),
        output_root=(args.output_root),
        canonical_width=(args.canonical_width),
        canonical_height=(args.canonical_height),
        docaligner_padding=(args.padding),
        min_overlap_ratio=(args.min_overlap),
        docaligner_model=(args.model),
    )

    # Models are initialized only once.
    pipeline = DrivingLicensePipeline(config)

    output_root = Path(config.output_root)

    for image_argument in args.images:
        image_path = Path(image_argument)

        # Separate directory for each image.
        output_dir = output_root / image_path.stem

        try:
            pipeline.process(
                image_path=(image_path),
                output_dir=(output_dir),
            )

        except Exception as error:
            print()
            print("=" * 70)
            print(f"FAILED: {image_path}")
            print("=" * 70)

            print(f"{type(error).__name__}: {error}")


if __name__ == "__main__":
    main()
