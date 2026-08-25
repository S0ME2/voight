import json
import os
import re

import cv2
from paddleocr import PaddleOCR


# =========================================================
# CONFIG
# =========================================================

IMAGE_PATH = "assets/samples/driving_license/test_license_canonical.jpg"
ROI_PATH = "config/driving_license/license_rois_canonical.json"

OUTPUT_DIR = "outputs/experiments/roi/rois"

CANONICAL_WIDTH = 1000
CANONICAL_HEIGHT = 630


# These are labels printed on the licence.
# We remove them if PaddleOCR joins the label and value
# into one OCR chunk.
#
# Example:
#   OCR -> "4a. 17.02.2026"
#   Result -> "17.02.2026"
FIELD_PREFIXES = {
    "surname": r"^1[\.\s]*",
    "given_names": r"^2[\.\s]*",
    "birth_place": r"^3[\.\s]*",
    "birth_date": r"^3[\.\s]*",
    "issue_date": r"^4a[\.\s]*",
    "expiry_date": r"^4b[\.\s]*",
    "issuer": r"^4c[\.\s]*",
    "personal_id": r"^4d[\.\s]*",
    "license_number": r"^5[\.\s]*",
    "address": r"^8[\.\s]*",
    "categories": r"^9[\.\s]*",
}


# =========================================================
# LOAD ROIS
# =========================================================


def load_rois(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# =========================================================
# OCR HELPERS
# =========================================================


def box_center(box):
    x1, y1, x2, y2 = box

    return (
        (x1 + x2) / 2,
        (y1 + y2) / 2,
    )


def normalized_roi_to_pixels(
    roi: dict,
    width: int,
    height: int,
):
    return (
        int(roi["x1"] * width),
        int(roi["y1"] * height),
        int(roi["x2"] * width),
        int(roi["y2"] * height),
    )


def center_inside_roi(
    box,
    roi_pixels,
) -> bool:
    center_x, center_y = box_center(box)

    x1, y1, x2, y2 = roi_pixels

    return x1 <= center_x <= x2 and y1 <= center_y <= y2


# =========================================================
# TEXT MERGING
# =========================================================


def merge_tokens(tokens: list[dict]) -> str:
    """
    Merge OCR chunks belonging to one field.

    Example:
        [
            {"text": "MIRZOHID"},
            {"text": "O'G'LI"}
        ]

    becomes:

        MIRZOHID O'G'LI
    """

    if not tokens:
        return ""

    # Sort mainly top-to-bottom, then left-to-right.
    tokens = sorted(
        tokens,
        key=lambda token: (
            token["center_y"],
            token["x1"],
        ),
    )

    return " ".join(token["text"].strip() for token in tokens if token["text"].strip())


def remove_field_prefix(
    field_name: str,
    value: str,
) -> str:
    pattern = FIELD_PREFIXES.get(field_name)

    if not pattern:
        return value.strip()

    return re.sub(
        pattern,
        "",
        value,
        flags=re.IGNORECASE,
    ).strip()


# =========================================================
# FIELD NORMALIZATION
# =========================================================


def normalize_spaces(value: str) -> str:
    return " ".join(value.split())


def normalize_date(value: str) -> str:
    """
    For now, only remove spaces.

    Proper date validation can be added later.
    """
    return value.replace(" ", "")


def normalize_id(value: str) -> str:
    return re.sub(
        r"[^A-Z0-9]",
        "",
        value.upper(),
    )


def normalize_field(
    field_name: str,
    value: str,
) -> str | None:
    value = remove_field_prefix(
        field_name,
        value,
    )

    if not value:
        return None

    if field_name in {
        "birth_date",
        "issue_date",
        "expiry_date",
    }:
        return normalize_date(value)

    if field_name in {
        "personal_id",
        "license_number",
    }:
        return normalize_id(value)

    return normalize_spaces(value)


# =========================================================
# MAIN EXTRACTION
# =========================================================


def extract_fields(
    result,
    rois: dict,
    image_width: int,
    image_height: int,
):
    texts = result["rec_texts"]
    scores = result["rec_scores"]
    boxes = result["rec_boxes"]

    ocr_tokens = []

    for text, score, box in zip(
        texts,
        scores,
        boxes,
    ):
        x1, y1, x2, y2 = map(
            float,
            box,
        )

        ocr_tokens.append(
            {
                "text": text,
                "score": float(score),
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_x": (x1 + x2) / 2,
                "center_y": (y1 + y2) / 2,
            }
        )

    extracted = {}
    matches = {}

    for field_name, roi in rois.items():
        roi_pixels = normalized_roi_to_pixels(
            roi,
            image_width,
            image_height,
        )

        field_tokens = []

        for token in ocr_tokens:
            box = (
                token["x1"],
                token["y1"],
                token["x2"],
                token["y2"],
            )

            if center_inside_roi(
                box,
                roi_pixels,
            ):
                field_tokens.append(token)

        raw_value = merge_tokens(field_tokens)

        value = normalize_field(
            field_name,
            raw_value,
        )

        extracted[field_name] = value
        matches[field_name] = field_tokens

    return extracted, matches


# =========================================================
# OUR ROI VISUALIZATION
# =========================================================


def save_roi_annotation(
    image,
    rois: dict,
    extracted: dict,
    matches: dict,
    output_path: str,
):
    annotated = image.copy()

    # Use actual image dimensions rather than
    # hardcoded dimensions.
    image_height, image_width = annotated.shape[:2]

    for field_name, roi in rois.items():
        x1, y1, x2, y2 = normalized_roi_to_pixels(
            roi,
            image_width,
            image_height,
        )

        # Draw our field ROI in green.
        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        # Draw OCR boxes assigned to this field in blue.
        for token in matches[field_name]:
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

        value = extracted.get(field_name)

        label = f"{field_name}: {value}" if value else f"{field_name}: <EMPTY>"

        # Draw extracted value in red.
        cv2.putText(
            annotated,
            label,
            (
                x1,
                max(y1 - 5, 15),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    success = cv2.imwrite(
        output_path,
        annotated,
    )

    if not success:
        raise RuntimeError(f"Could not save ROI annotation: {output_path}")


# =========================================================
# RUN
# =========================================================


def main():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # ---------------------------------------------
    # 1. Load already-canonical image
    # ---------------------------------------------
    #
    # IMPORTANT:
    #
    # IMAGE_PATH must point to an image that has
    # ALREADY been:
    #
    #   1. Cropped to the physical licence
    #   2. Perspective-corrected
    #   3. Warped to 1000x630
    #
    # We DO NOT resize the image here.
    # ---------------------------------------------

    canonical = cv2.imread(IMAGE_PATH)

    if canonical is None:
        raise FileNotFoundError(f"Could not load: {IMAGE_PATH}")

    image_height, image_width = canonical.shape[:2]

    # ---------------------------------------------
    # 2. Verify canonical dimensions
    # ---------------------------------------------

    if image_width != CANONICAL_WIDTH or image_height != CANONICAL_HEIGHT:
        raise ValueError(
            "\n"
            "Canonical image has incorrect dimensions.\n"
            "\n"
            f"Expected: "
            f"{CANONICAL_WIDTH}x{CANONICAL_HEIGHT}\n"
            f"Got: "
            f"{image_width}x{image_height}\n"
            "\n"
            "The input image must first go through "
            "perspective canonicalization."
        )

    print()
    print("=" * 60)
    print("CANONICAL IMAGE")
    print("=" * 60)
    print(f"Input: {IMAGE_PATH}")
    print(f"Size: {image_width}x{image_height}")

    # ---------------------------------------------
    # 3. Save exact image used by OCR
    # ---------------------------------------------
    #
    # This lets us verify exactly what PaddleOCR
    # and our ROI extractor received.
    # ---------------------------------------------

    canonical_path = os.path.join(
        OUTPUT_DIR,
        "canonical_input.jpg",
    )

    success = cv2.imwrite(
        canonical_path,
        canonical,
    )

    if not success:
        raise RuntimeError(f"Could not save: {canonical_path}")

    # ---------------------------------------------
    # 4. Load ROIs
    # ---------------------------------------------

    rois = load_rois(ROI_PATH)

    print(f"Loaded {len(rois)} ROIs from: {ROI_PATH}")

    # ---------------------------------------------
    # 5. Initialize PaddleOCR
    # ---------------------------------------------

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    # ---------------------------------------------
    # 6. Run PaddleOCR on canonical image
    # ---------------------------------------------

    results = ocr.predict(canonical_path)

    # ---------------------------------------------
    # 7. Process OCR results
    # ---------------------------------------------

    for result_index, result in enumerate(results):
        # -----------------------------------------
        # 7.1 Save PaddleOCR annotated image
        # -----------------------------------------

        if result_index == 0:
            paddle_annotation_path = os.path.join(
                OUTPUT_DIR,
                "paddle_ocr_annotated.jpg",
            )
        else:
            paddle_annotation_path = os.path.join(
                OUTPUT_DIR,
                f"paddle_ocr_annotated_{result_index}.jpg",
            )

        result.save_to_img(paddle_annotation_path)

        # -----------------------------------------
        # 7.2 Extract fields using our ROIs
        # -----------------------------------------

        extracted, matches = extract_fields(
            result=result,
            rois=rois,
            image_width=image_width,
            image_height=image_height,
        )

        # -----------------------------------------
        # 7.3 Save our ROI annotated image
        # -----------------------------------------

        if result_index == 0:
            roi_annotation_path = os.path.join(
                OUTPUT_DIR,
                "roi_annotated.jpg",
            )
        else:
            roi_annotation_path = os.path.join(
                OUTPUT_DIR,
                f"roi_annotated_{result_index}.jpg",
            )

        save_roi_annotation(
            image=canonical,
            rois=rois,
            extracted=extracted,
            matches=matches,
            output_path=roi_annotation_path,
        )

        # -----------------------------------------
        # 7.4 Save extracted JSON
        # -----------------------------------------

        if result_index == 0:
            json_path = os.path.join(
                OUTPUT_DIR,
                "extracted.json",
            )
        else:
            json_path = os.path.join(
                OUTPUT_DIR,
                f"extracted_{result_index}.json",
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
        # 7.5 Print extraction result
        # -----------------------------------------

        print()
        print("=" * 60)
        print("EXTRACTED FIELDS")
        print("=" * 60)

        print(
            json.dumps(
                extracted,
                ensure_ascii=False,
                indent=4,
            )
        )

        # -----------------------------------------
        # 7.6 Print matched OCR chunks
        # -----------------------------------------
        #
        # This is useful for debugging cases where
        # a field is incorrect.
        # -----------------------------------------

        print()
        print("=" * 60)
        print("OCR TOKENS MATCHED TO EACH FIELD")
        print("=" * 60)

        for field_name, field_tokens in matches.items():
            print()
            print(f"{field_name}:")

            if not field_tokens:
                print("  <NO OCR TOKENS>")
                continue

            for token in field_tokens:
                print(
                    f"  "
                    f"{token['text']!r} "
                    f"(score={token['score']:.3f}, "
                    f"center="
                    f"({token['center_x']:.1f}, "
                    f"{token['center_y']:.1f}))"
                )

        # -----------------------------------------
        # 7.7 Print output paths
        # -----------------------------------------

        print()
        print("=" * 60)
        print("OUTPUT FILES")
        print("=" * 60)

        print(f"Canonical input:       {canonical_path}")

        print(f"PaddleOCR annotation: {paddle_annotation_path}")

        print(f"ROI annotation:       {roi_annotation_path}")

        print(f"Extracted JSON:       {json_path}")


if __name__ == "__main__":
    main()
