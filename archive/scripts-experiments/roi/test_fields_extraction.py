import json
import os
import re

import cv2
from paddleocr import PaddleOCR


# =========================================================
# CONFIG
# =========================================================

IMAGE_PATH = "assets/samples/driving_license/test_license_data_crop.jpg"
ROI_PATH = "config/driving_license/field_rois_crop.json"

OUTPUT_DIR = "outputs/experiments/roi/fields_extraction"

# Minimum percentage of an OCR box that should overlap
# a field ROI before we consider it a candidate.
MIN_OVERLAP_RATIO = 0.30


# =========================================================
# FIELD PREFIXES
# =========================================================
#
# These are NOT used to determine which field the text
# belongs to.
#
# Position / ROI is the primary signal.
#
# These regexes are only used AFTER field assignment
# to remove labels when OCR recognizes them.
#
# Examples:
#
#   "1. QOBULOV"       -> "QOBULOV"
#   "2 HUSNIDDIN"      -> "HUSNIDDIN"
#   "4a. 17.02.2026"   -> "17.02.2026"
#
# If OCR misses the label completely, that's fine.
# =========================================================

FIELD_PREFIXES = {
    "surname": [
        r"^\s*1[\.\s:,-]*",
    ],
    "given_names": [
        r"^\s*2[\.\s:,-]*",
    ],
    "birth_place_and_date": [
        r"^\s*3[\.\s:,-]*",
    ],
    "issue_date": [
        r"^\s*4\s*[aA][\.\s:,-]*",
    ],
    "expiry_date": [
        r"^\s*4\s*[bB][\.\s:,-]*",
    ],
    "issued_place": [
        r"^\s*4\s*[cC][\.\s:,-]*",
    ],
    "personal_id": [
        r"^\s*4\s*[dD][\.\s:,-]*",
    ],
    "license_number": [
        r"^\s*5[\.\s:,-]*",
    ],
    "address": [
        r"^\s*8[\.\s:,-]*",
    ],
    "categories": [
        r"^\s*9[\.\s:,-]*",
    ],
}


# =========================================================
# LOAD DATA
# =========================================================


def load_rois(path: str) -> dict:
    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


# =========================================================
# ROI HELPERS
# =========================================================


def normalized_roi_to_pixels(
    roi: dict,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:

    return (
        roi["x1"] * image_width,
        roi["y1"] * image_height,
        roi["x2"] * image_width,
        roi["y2"] * image_height,
    )


def box_area(
    box: tuple[float, float, float, float],
) -> float:

    x1, y1, x2, y2 = box

    width = max(
        0.0,
        x2 - x1,
    )

    height = max(
        0.0,
        y2 - y1,
    )

    return width * height


def intersection_area(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
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
    token_box: tuple[float, float, float, float],
    roi_box: tuple[float, float, float, float],
) -> float:
    """
    Returns how much of the OCR TOKEN is inside the ROI.

    Example:

    OCR token area = 100 pixels
    80 pixels are inside ROI

    overlap_ratio = 0.80

    We use OCR-token area as denominator because
    our question is:

        "How much of this OCR result belongs to this ROI?"
    """

    token_area = box_area(token_box)

    if token_area <= 0:
        return 0.0

    overlap = intersection_area(
        token_box,
        roi_box,
    )

    return overlap / token_area


# =========================================================
# OCR TOKEN CREATION
# =========================================================


def create_ocr_tokens(
    result,
) -> list[dict]:

    texts = result["rec_texts"]
    scores = result["rec_scores"]
    boxes = result["rec_boxes"]

    tokens = []

    for text, score, box in zip(
        texts,
        scores,
        boxes,
    ):
        x1, y1, x2, y2 = map(
            float,
            box,
        )

        tokens.append(
            {
                "text": text.strip(),
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


# =========================================================
# ASSIGN OCR TOKENS TO FIELD ROIS
# =========================================================


def assign_tokens_to_fields(
    tokens: list[dict],
    rois: dict,
    image_width: int,
    image_height: int,
) -> tuple[dict, list]:

    # Convert normalized ROIs to pixel coordinates once.
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

        # -----------------------------------------
        # Compare this OCR token with every ROI.
        # -----------------------------------------

        for field_name, roi_box in pixel_rois.items():
            overlap = token_roi_overlap_ratio(
                token_box,
                roi_box,
            )

            if overlap > best_overlap:
                best_overlap = overlap
                best_field = field_name

        # -----------------------------------------
        # Assign token only to the BEST ROI.
        #
        # This avoids duplicate assignment when
        # generous ROIs slightly overlap.
        # -----------------------------------------

        if best_field is not None and best_overlap >= MIN_OVERLAP_RATIO:
            token_with_overlap = {
                **token,
                "overlap_ratio": best_overlap,
            }

            assignments[best_field].append(token_with_overlap)

        else:
            unassigned.append(token)

    return assignments, unassigned


# =========================================================
# MERGE TOKENS
# =========================================================


def merge_tokens(
    tokens: list[dict],
) -> str:
    """
    Merge OCR chunks in reading order.

    Example:

        "2 HUSNIDDIN"
        "MIRZOHID O'G'LI"

    becomes:

        "2 HUSNIDDIN MIRZOHID O'G'LI"
    """

    if not tokens:
        return ""

    # Primarily top-to-bottom,
    # secondarily left-to-right.
    sorted_tokens = sorted(
        tokens,
        key=lambda token: (
            token["center_y"],
            token["x1"],
        ),
    )

    return " ".join(token["text"] for token in sorted_tokens if token["text"])


# =========================================================
# PREFIX REMOVAL
# =========================================================


def remove_field_prefix(
    field_name: str,
    value: str,
) -> str:

    patterns = FIELD_PREFIXES.get(
        field_name,
        [],
    )

    cleaned = value.strip()

    for pattern in patterns:
        cleaned = re.sub(
            pattern,
            "",
            cleaned,
            count=1,
            flags=re.IGNORECASE,
        )

    return cleaned.strip()


# =========================================================
# GENERAL NORMALIZATION
# =========================================================


def normalize_spaces(
    value: str,
) -> str:

    return " ".join(value.split())


# =========================================================
# DATE HELPERS
# =========================================================


DATE_PATTERN = re.compile(
    r"\b"
    r"(\d{1,2})"
    r"[.\-/]"
    r"(\d{1,2})"
    r"[.\-/]"
    r"(\d{4})"
    r"\b"
)


def extract_date(
    value: str,
) -> str | None:

    match = DATE_PATTERN.search(value)

    if not match:
        return None

    day, month, year = match.groups()

    return f"{int(day):02d}.{int(month):02d}.{year}"


# =========================================================
# FIELD PROCESSORS
# =========================================================


def process_simple_text(
    field_name: str,
    value: str,
) -> str | None:

    value = remove_field_prefix(
        field_name,
        value,
    )

    value = normalize_spaces(value)

    return value or None


def process_date(
    field_name: str,
    value: str,
) -> str | None:

    value = remove_field_prefix(
        field_name,
        value,
    )

    return extract_date(value)


def process_personal_id(
    value: str,
) -> str | None:

    value = remove_field_prefix(
        "personal_id",
        value,
    )

    # Personal ID should be numeric.
    digits = re.sub(
        r"\D",
        "",
        value,
    )

    return digits or None


def process_license_number(
    value: str,
) -> str | None:

    value = remove_field_prefix(
        "license_number",
        value,
    )

    cleaned = re.sub(
        r"[^A-Z0-9]",
        "",
        value.upper(),
    )

    return cleaned or None


def process_categories(
    value: str,
) -> str | None:

    value = remove_field_prefix(
        "categories",
        value,
    )

    value = normalize_spaces(value.upper())

    return value or None


def process_serial_number(
    value: str,
) -> str | None:

    cleaned = re.sub(
        r"[^A-Z0-9]",
        "",
        value.upper(),
    )

    return cleaned or None


def process_birth_place_and_date(
    value: str,
) -> tuple[str | None, str | None]:

    value = remove_field_prefix(
        "birth_place_and_date",
        value,
    )

    date_match = DATE_PATTERN.search(value)

    if not date_match:
        return (
            normalize_spaces(value) or None,
            None,
        )

    day, month, year = date_match.groups()

    birth_date = f"{int(day):02d}.{int(month):02d}.{year}"

    # Everything before the date
    # becomes birth place.
    birth_place = value[: date_match.start()]

    birth_place = normalize_spaces(birth_place)

    return (
        birth_place or None,
        birth_date,
    )


# =========================================================
# CREATE FINAL JSON
# =========================================================


def build_extracted_data(
    assignments: dict,
) -> dict:

    # First merge OCR tokens for each ROI.
    raw_fields = {}

    for field_name, tokens in assignments.items():
        raw_fields[field_name] = merge_tokens(tokens)

    # ---------------------------------------------
    # Birth place + birth date
    # ---------------------------------------------

    birth_place, birth_date = process_birth_place_and_date(
        raw_fields.get(
            "birth_place_and_date",
            "",
        )
    )

    # ---------------------------------------------
    # Final structured result
    # ---------------------------------------------

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
        "issue_date": process_date(
            "issue_date",
            raw_fields.get(
                "issue_date",
                "",
            ),
        ),
        "expiry_date": process_date(
            "expiry_date",
            raw_fields.get(
                "expiry_date",
                "",
            ),
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
        "license_number": process_license_number(
            raw_fields.get(
                "license_number",
                "",
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
        "serial_number": process_serial_number(
            raw_fields.get(
                "serial_number",
                "",
            )
        ),
    }

    return extracted, raw_fields


# =========================================================
# ANNOTATION
# =========================================================


def save_debug_annotation(
    image,
    rois: dict,
    assignments: dict,
    output_path: str,
):

    annotated = image.copy()

    image_height, image_width = image.shape[:2]

    for field_name, roi in rois.items():
        x1, y1, x2, y2 = normalized_roi_to_pixels(
            roi,
            image_width,
            image_height,
        )

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

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

        # Blue = OCR token assigned
        # to this specific field.
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

    cv2.imwrite(
        output_path,
        annotated,
    )


# =========================================================
# MAIN
# =========================================================


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # ---------------------------------------------
    # Load cropped data image
    # ---------------------------------------------

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(f"Could not load: {IMAGE_PATH}")

    image_height, image_width = image.shape[:2]

    print()
    print(f"Image size: {image_width}x{image_height}")

    # ---------------------------------------------
    # Load field ROIs
    # ---------------------------------------------

    rois = load_rois(ROI_PATH)

    print(f"Loaded {len(rois)} field ROIs.")

    # ---------------------------------------------
    # Initialize OCR
    # ---------------------------------------------

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    # ---------------------------------------------
    # Run OCR ONCE
    # ---------------------------------------------

    results = ocr.predict(IMAGE_PATH)

    for result in results:
        # -----------------------------------------
        # Save PaddleOCR annotation
        # -----------------------------------------

        paddle_path = os.path.join(
            OUTPUT_DIR,
            "paddle_ocr_annotated.jpg",
        )

        result.save_to_img(paddle_path)

        # -----------------------------------------
        # Create OCR tokens
        # -----------------------------------------

        tokens = create_ocr_tokens(result)

        # -----------------------------------------
        # Assign tokens to ROIs
        # -----------------------------------------

        assignments, unassigned = assign_tokens_to_fields(
            tokens=tokens,
            rois=rois,
            image_width=image_width,
            image_height=image_height,
        )

        # -----------------------------------------
        # Build clean JSON
        # -----------------------------------------

        extracted, raw_fields = build_extracted_data(assignments)

        # -----------------------------------------
        # Save final JSON
        # -----------------------------------------

        json_path = os.path.join(
            OUTPUT_DIR,
            "extracted.json",
        )

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                extracted,
                file,
                ensure_ascii=False,
                indent=4,
            )

        # -----------------------------------------
        # Save raw field values
        #
        # Very useful while debugging.
        # -----------------------------------------

        raw_json_path = os.path.join(
            OUTPUT_DIR,
            "raw_fields.json",
        )

        with open(
            raw_json_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                raw_fields,
                file,
                ensure_ascii=False,
                indent=4,
            )

        # -----------------------------------------
        # Save our ROI/token annotation
        # -----------------------------------------

        debug_path = os.path.join(
            OUTPUT_DIR,
            "field_assignment_annotated.jpg",
        )

        save_debug_annotation(
            image=image,
            rois=rois,
            assignments=assignments,
            output_path=debug_path,
        )

        # -----------------------------------------
        # Print result
        # -----------------------------------------

        print()
        print("=" * 70)
        print("RAW FIELDS")
        print("=" * 70)

        print(
            json.dumps(
                raw_fields,
                ensure_ascii=False,
                indent=4,
            )
        )

        print()
        print("=" * 70)
        print("FINAL EXTRACTED JSON")
        print("=" * 70)

        print(
            json.dumps(
                extracted,
                ensure_ascii=False,
                indent=4,
            )
        )

        # -----------------------------------------
        # Print unassigned OCR results
        # -----------------------------------------

        print()
        print("=" * 70)
        print("UNASSIGNED OCR TOKENS")
        print("=" * 70)

        if not unassigned:
            print("All OCR tokens were assigned.")

        else:
            for token in unassigned:
                print(f"{token['text']!r} (score={token['score']:.3f})")

        # -----------------------------------------
        # Output paths
        # -----------------------------------------

        print()
        print("=" * 70)
        print("OUTPUT")
        print("=" * 70)

        print(f"PaddleOCR annotation: {paddle_path}")

        print(f"Field assignment:     {debug_path}")

        print(f"Raw fields:           {raw_json_path}")

        print(f"Final JSON:           {json_path}")


if __name__ == "__main__":
    main()
