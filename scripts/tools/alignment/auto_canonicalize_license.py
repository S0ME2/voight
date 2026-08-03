import os

import cv2
import numpy as np


# =========================================================
# CONFIG
# =========================================================

IMAGE_PATH = "assets/samples/driving_license/test_license.jpg"

OUTPUT_DIR = "outputs/experiments/alignment/auto_canonicalize"

CANONICAL_WIDTH = 1000
CANONICAL_HEIGHT = 630

# Our expected canonical aspect ratio.
TARGET_ASPECT_RATIO = CANONICAL_WIDTH / CANONICAL_HEIGHT

# Ignore very small contours.
MIN_DOCUMENT_AREA_RATIO = 0.15


# =========================================================
# POINT ORDERING
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

    points = points.astype(np.float32)

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


# =========================================================
# QUADRILATERAL HELPERS
# =========================================================


def distance(
    point_a: np.ndarray,
    point_b: np.ndarray,
) -> float:

    return float(np.linalg.norm(point_a - point_b))


def quadrilateral_aspect_ratio(
    points: np.ndarray,
) -> float:
    """
    Estimate the aspect ratio of a detected
    quadrilateral.

    Uses the average top/bottom width and
    left/right height.
    """

    tl, tr, br, bl = order_points(points)

    top_width = distance(
        tl,
        tr,
    )

    bottom_width = distance(
        bl,
        br,
    )

    left_height = distance(
        tl,
        bl,
    )

    right_height = distance(
        tr,
        br,
    )

    average_width = (top_width + bottom_width) / 2

    average_height = (left_height + right_height) / 2

    if average_height == 0:
        return 0.0

    ratio = average_width / average_height

    # Handle 90-degree orientation.
    if ratio < 1:
        ratio = 1 / ratio

    return ratio


# =========================================================
# DOCUMENT DETECTION
# =========================================================


def detect_document(
    image: np.ndarray,
):
    """
    Try to detect the driving licence boundary.

    Returns:

        corners,
        edges,
        debug_contours

    corners is None when detection fails.
    """

    image_height, image_width = image.shape[:2]

    image_area = image_width * image_height

    # ---------------------------------------------
    # 1. Grayscale
    # ---------------------------------------------

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # ---------------------------------------------
    # 2. Blur
    #
    # Reduces small texture/noise before Canny.
    # ---------------------------------------------

    blurred = cv2.GaussianBlur(
        gray,
        (5, 5),
        0,
    )

    # ---------------------------------------------
    # 3. Detect edges
    # ---------------------------------------------

    edges = cv2.Canny(
        blurred,
        50,
        150,
    )

    # ---------------------------------------------
    # 4. Close small gaps in licence border
    # ---------------------------------------------

    kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (5, 5),
    )

    edges_closed = cv2.morphologyEx(
        edges,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2,
    )

    # ---------------------------------------------
    # 5. Find external contours
    # ---------------------------------------------

    contours, _ = cv2.findContours(
        edges_closed,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    # Largest first.
    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True,
    )

    debug = image.copy()

    best_corners = None
    best_score = float("-inf")

    # ---------------------------------------------
    # 6. Inspect largest contours
    # ---------------------------------------------

    for contour in contours[:20]:
        contour_area = cv2.contourArea(contour)

        area_ratio = contour_area / image_area

        # Licence should occupy a meaningful
        # portion of the photograph.
        if area_ratio < MIN_DOCUMENT_AREA_RATIO:
            continue

        perimeter = cv2.arcLength(
            contour,
            True,
        )

        # Approximate contour as polygon.
        polygon = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True,
        )

        # We want exactly four corners.
        if len(polygon) != 4:
            continue

        if not cv2.isContourConvex(polygon):
            continue

        corners = polygon.reshape(
            4,
            2,
        ).astype(np.float32)

        aspect_ratio = quadrilateral_aspect_ratio(corners)

        # -----------------------------------------
        # Score candidate
        #
        # Prefer:
        # 1. Large contours
        # 2. Aspect ratio close to canonical ratio
        # -----------------------------------------

        aspect_error = abs(aspect_ratio - TARGET_ASPECT_RATIO)

        score = area_ratio * 10 - aspect_error

        # Draw all accepted candidate quadrilaterals
        # in yellow for debugging.
        cv2.polylines(
            debug,
            [corners.astype(np.int32)],
            True,
            (0, 255, 255),
            2,
        )

        if score > best_score:
            best_score = score
            best_corners = corners

    # ---------------------------------------------
    # Draw selected document
    # ---------------------------------------------

    if best_corners is not None:
        cv2.polylines(
            debug,
            [best_corners.astype(np.int32)],
            True,
            (0, 255, 0),
            5,
        )

        ordered = order_points(best_corners)

        labels = [
            "TL",
            "TR",
            "BR",
            "BL",
        ]

        for point, label in zip(
            ordered,
            labels,
        ):
            x, y = map(
                int,
                point,
            )

            cv2.circle(
                debug,
                (x, y),
                7,
                (0, 0, 255),
                -1,
            )

            cv2.putText(
                debug,
                label,
                (
                    x + 10,
                    y,
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

    return (
        best_corners,
        edges_closed,
        debug,
    )


# =========================================================
# PERSPECTIVE WARP
# =========================================================


def warp_document(
    image: np.ndarray,
    corners: np.ndarray,
) -> np.ndarray:
    """
    Warp detected licence to fixed canonical size.
    """

    source = order_points(corners)

    destination = np.array(
        [
            [
                0,
                0,
            ],
            [
                CANONICAL_WIDTH - 1,
                0,
            ],
            [
                CANONICAL_WIDTH - 1,
                CANONICAL_HEIGHT - 1,
            ],
            [
                0,
                CANONICAL_HEIGHT - 1,
            ],
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
            CANONICAL_WIDTH,
            CANONICAL_HEIGHT,
        ),
    )

    return canonical


# =========================================================
# MAIN
# =========================================================


def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # ---------------------------------------------
    # Load original photograph
    # ---------------------------------------------

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(f"Could not load: {IMAGE_PATH}")

    # ---------------------------------------------
    # Detect licence
    # ---------------------------------------------

    (
        corners,
        edges,
        detection_debug,
    ) = detect_document(image)

    # ---------------------------------------------
    # Save edge-detection debug image
    # ---------------------------------------------

    edges_path = os.path.join(
        OUTPUT_DIR,
        "01_edges.jpg",
    )

    cv2.imwrite(
        edges_path,
        edges,
    )

    # ---------------------------------------------
    # Save document detection visualization
    # ---------------------------------------------

    detection_path = os.path.join(
        OUTPUT_DIR,
        "02_document_detection.jpg",
    )

    cv2.imwrite(
        detection_path,
        detection_debug,
    )

    # ---------------------------------------------
    # Fail clearly when no licence is found
    # ---------------------------------------------

    if corners is None:
        print()
        print("=" * 60)
        print("DOCUMENT DETECTION FAILED")
        print("=" * 60)

        print("No suitable 4-corner document was detected.")

        print()
        print(f"Inspect edges: {edges_path}")

        print(f"Inspect detection: {detection_path}")

        return

    # ---------------------------------------------
    # Perspective warp
    # ---------------------------------------------

    canonical = warp_document(
        image,
        corners,
    )

    canonical_path = os.path.join(
        OUTPUT_DIR,
        "03_license_canonical.jpg",
    )

    cv2.imwrite(
        canonical_path,
        canonical,
    )

    # ---------------------------------------------
    # Output
    # ---------------------------------------------

    print()
    print("=" * 60)
    print("DOCUMENT DETECTED")
    print("=" * 60)

    print(f"Edges:      {edges_path}")

    print(f"Detection:  {detection_path}")

    print(f"Canonical:  {canonical_path}")

    print()

    print("Detected corners:")

    for label, point in zip(
        [
            "TL",
            "TR",
            "BR",
            "BL",
        ],
        order_points(corners),
    ):
        print(f"  {label}: ({point[0]:.1f}, {point[1]:.1f})")


if __name__ == "__main__":
    main()
