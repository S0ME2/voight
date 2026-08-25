from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import cv2
import numpy as np
from mrzscanner import MRZScanner, ModelType
from paddleocr import PaddleOCR


# =========================================================
# DEFAULT CONFIG
# =========================================================

DEFAULT_IMAGE = Path("assets/samples/id_card/realidcard.png")

DEFAULT_OUTPUT_ROOT = Path("outputs/experiments/mrz/mrzscanner")

# Slightly expand the polygon predicted by MRZScanner
# before cropping so that edge characters are less likely
# to be accidentally cut off.
POLYGON_PADDING_RATIO = 0.03


# =========================================================
# GENERAL HELPERS
# =========================================================


def save_json(
    path: Path,
    data,
) -> None:
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
# POLYGON HELPERS
# =========================================================


def order_points(
    points: np.ndarray,
) -> np.ndarray:
    """
    Order four points as:

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

    ordered[0] = points[np.argmin(point_sum)]

    ordered[2] = points[np.argmax(point_sum)]

    ordered[1] = points[np.argmin(point_diff)]

    ordered[3] = points[np.argmax(point_diff)]

    return ordered


def expand_polygon(
    polygon: np.ndarray,
    image_width: int,
    image_height: int,
    ratio: float,
) -> np.ndarray:
    """
    Expand a quadrilateral away from its center.

    This adds a small margin around the detected MRZ.
    """

    polygon = np.asarray(
        polygon,
        dtype=np.float32,
    ).reshape(4, 2)

    center = polygon.mean(axis=0)

    expanded = center + (polygon - center) * (1.0 + ratio)

    expanded[:, 0] = np.clip(
        expanded[:, 0],
        0,
        image_width - 1,
    )

    expanded[:, 1] = np.clip(
        expanded[:, 1],
        0,
        image_height - 1,
    )

    return expanded


def warp_polygon(
    image: np.ndarray,
    polygon: np.ndarray,
) -> np.ndarray:
    """
    Perspective-correct the detected MRZ polygon.

    The output dimensions are calculated dynamically
    from the detected quadrilateral.
    """

    tl, tr, br, bl = order_points(polygon)

    top_width = np.linalg.norm(tr - tl)

    bottom_width = np.linalg.norm(br - bl)

    left_height = np.linalg.norm(bl - tl)

    right_height = np.linalg.norm(br - tr)

    output_width = max(
        1,
        int(
            round(
                max(
                    top_width,
                    bottom_width,
                )
            )
        ),
    )

    output_height = max(
        1,
        int(
            round(
                max(
                    left_height,
                    right_height,
                )
            )
        ),
    )

    destination = np.array(
        [
            [0, 0],
            [
                output_width - 1,
                0,
            ],
            [
                output_width - 1,
                output_height - 1,
            ],
            [
                0,
                output_height - 1,
            ],
        ],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(
        np.array(
            [
                tl,
                tr,
                br,
                bl,
            ],
            dtype=np.float32,
        ),
        destination,
    )

    return cv2.warpPerspective(
        image,
        transform,
        (
            output_width,
            output_height,
        ),
    )


def draw_detection(
    image: np.ndarray,
    polygon: np.ndarray,
) -> np.ndarray:
    output = image.copy()

    ordered = order_points(polygon)

    cv2.polylines(
        output,
        [ordered.astype(np.int32)],
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
            7,
            (0, 0, 255),
            -1,
        )

        cv2.putText(
            output,
            label,
            (
                x + 8,
                max(
                    20,
                    y - 8,
                ),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return output


# =========================================================
# PADDLEOCR HELPERS
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
                "text": str(text).strip(),
                "score": float(score),
                "box": [
                    x1,
                    y1,
                    x2,
                    y2,
                ],
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
                "center_y": (y1 + y2) / 2,
                "height": (y2 - y1),
            }
        )

    return tokens


def normalize_mrz_fragment(
    text: str,
) -> str:
    """
    Keep only characters valid in an MRZ.

    Spaces and ordinary punctuation generated by OCR
    are discarded.
    """

    return re.sub(
        r"[^A-Z0-9<]",
        "",
        text.upper(),
    )


def reconstruct_mrz_lines(
    tokens: list[dict],
) -> list[str]:
    """
    Group OCR tokens into lines using their vertical
    positions, then concatenate tokens left-to-right.

    This also works if PaddleOCR splits one MRZ line
    into several OCR boxes.
    """

    if not tokens:
        return []

    useful_tokens = [token for token in tokens if normalize_mrz_fragment(token["text"])]

    if not useful_tokens:
        return []

    heights = [token["height"] for token in useful_tokens if token["height"] > 0]

    median_height = float(np.median(heights)) if heights else 10.0

    line_threshold = max(
        5.0,
        median_height * 0.7,
    )

    sorted_tokens = sorted(
        useful_tokens,
        key=lambda token: (
            token["center_y"],
            token["x1"],
        ),
    )

    lines: list[list[dict]] = []

    for token in sorted_tokens:
        best_line = None
        best_distance = float("inf")

        for line in lines:
            average_y = sum(item["center_y"] for item in line) / len(line)

            distance = abs(token["center_y"] - average_y)

            if distance <= line_threshold and distance < best_distance:
                best_line = line
                best_distance = distance

        if best_line is None:
            lines.append([token])
        else:
            best_line.append(token)

    lines.sort(key=lambda line: sum(token["center_y"] for token in line) / len(line))

    reconstructed = []

    for line in lines:
        line.sort(key=lambda token: token["x1"])

        text = "".join(normalize_mrz_fragment(token["text"]) for token in line)

        # Ignore obvious tiny OCR noise.
        if len(text) >= 20:
            reconstructed.append(text)

    return reconstructed


# =========================================================
# ARGUMENTS
# =========================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=("Test MRZScanner detection + PaddleOCR recognition.")
    )

    parser.add_argument(
        "image",
        nargs="?",
        default=str(DEFAULT_IMAGE),
    )

    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
    )

    return parser.parse_args()


# =========================================================
# MAIN
# =========================================================


def main():
    args = parse_args()

    image_path = Path(args.image)

    output_dir = Path(args.output_root) / image_path.stem

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not load: {image_path}")

    image_height, image_width = image.shape[:2]

    total_start = time.perf_counter()

    # =====================================================
    # MODEL INITIALIZATION
    # =====================================================

    start = time.perf_counter()

    detector = MRZScanner(
        model_type=(ModelType.detection),
        detection_cfg="20250222",
    )

    detector_init_seconds = time.perf_counter() - start

    start = time.perf_counter()

    # Replace this initialization later with your exact
    # best PaddleOCR configuration if best_run.py differs.
    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    paddle_init_seconds = time.perf_counter() - start

    processing_start = time.perf_counter()

    # =====================================================
    # MRZ DETECTION
    # =====================================================

    start = time.perf_counter()

    detector_result = detector(
        image,
        do_center_crop=False,
    )

    detection_seconds = time.perf_counter() - start

    polygon = detector_result.get("mrz_polygon")

    if polygon is None:
        raise RuntimeError("MRZScanner did not detect an MRZ region.")

    polygon = np.asarray(
        polygon,
        dtype=np.float32,
    ).reshape(
        4,
        2,
    )

    # Save original detector polygon.
    save_json(
        output_dir / "detector_result.json",
        {
            "mrz_polygon": (polygon.tolist()),
            "msg": str(detector_result.get("msg")),
        },
    )

    detection_image = draw_detection(
        image,
        polygon,
    )

    cv2.imwrite(
        str(output_dir / "01_mrz_detection.jpg"),
        detection_image,
    )

    # =====================================================
    # MRZ CROP
    # =====================================================

    start = time.perf_counter()

    expanded_polygon = expand_polygon(
        polygon,
        image_width,
        image_height,
        POLYGON_PADDING_RATIO,
    )

    mrz_crop = warp_polygon(
        image,
        expanded_polygon,
    )

    crop_seconds = time.perf_counter() - start

    mrz_crop_path = output_dir / "02_mrz_crop.jpg"

    if not cv2.imwrite(
        str(mrz_crop_path),
        mrz_crop,
    ):
        raise RuntimeError("Failed to save MRZ crop.")

    # =====================================================
    # PADDLE OCR
    # =====================================================

    start = time.perf_counter()

    ocr_results = list(ocr.predict(str(mrz_crop_path)))

    ocr_seconds = time.perf_counter() - start

    if not ocr_results:
        raise RuntimeError("PaddleOCR returned no result.")

    result = ocr_results[0]

    paddle_annotation_path = output_dir / "03_paddle_ocr_annotated.jpg"

    result.save_to_img(str(paddle_annotation_path))

    # =====================================================
    # MRZ RECONSTRUCTION
    # =====================================================

    start = time.perf_counter()

    tokens = create_ocr_tokens(result)

    mrz_lines = reconstruct_mrz_lines(tokens)

    reconstruction_seconds = time.perf_counter() - start

    mrz_text = "\n".join(mrz_lines)

    save_json(
        output_dir / "raw_ocr.json",
        tokens,
    )

    save_json(
        output_dir / "result.json",
        {
            "mrz_lines": (mrz_lines),
            "mrz_text": (mrz_text),
        },
    )

    processing_seconds = time.perf_counter() - processing_start

    total_seconds = time.perf_counter() - total_start

    timings = {
        "detector_init_seconds": (detector_init_seconds),
        "paddle_init_seconds": (paddle_init_seconds),
        "detection_seconds": (detection_seconds),
        "crop_seconds": (crop_seconds),
        "ocr_seconds": (ocr_seconds),
        "reconstruction_seconds": (reconstruction_seconds),
        # Best number for comparison after models
        # have already been downloaded/cached.
        "processing_seconds": (processing_seconds),
        # Includes model initialization.
        "total_seconds": (total_seconds),
    }

    save_json(
        output_dir / "timings.json",
        timings,
    )

    print()
    print("=" * 70)
    print("MRZSCANNER + PADDLEOCR")
    print("=" * 70)

    print()
    print("MRZ:")

    print(mrz_text or "<NO MRZ TEXT>")

    print()
    print(
        json.dumps(
            timings,
            indent=4,
        )
    )

    print()
    print(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
