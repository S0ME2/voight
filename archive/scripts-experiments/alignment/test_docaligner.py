import os

import cv2
import numpy as np
from docaligner import DocAligner


# =========================================================
# CONFIG
# =========================================================

IMAGE_PATH = "assets/samples/id_card/inhand.jpg"

OUTPUT_DIR = "outputs/experiments/alignment/docaligner/id_card"

CANONICAL_WIDTH = 1000
CANONICAL_HEIGHT = 630  # 704 # 630 for id-cards

# Padding helps when the licence is very close
# to the edge of the original photograph.
PADDING = 100


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


# =========================================================
# DRAW DETECTED CORNERS
# =========================================================


def draw_polygon(
    image: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:

    output = image.copy()

    ordered = order_points(points)

    labels = [
        "TL",
        "TR",
        "BR",
        "BL",
    ]

    # Draw polygon.
    polygon = ordered.astype(np.int32)

    cv2.polylines(
        output,
        [polygon],
        isClosed=True,
        color=(0, 255, 0),
        thickness=4,
    )

    # Draw corner points.
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
                y - 10,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    return output


# =========================================================
# PERSPECTIVE WARP
# =========================================================


def warp_to_canonical(
    image: np.ndarray,
    points: np.ndarray,
) -> np.ndarray:
    """
    Warp detected licence into fixed:

        1000 x 630 or 704

    canonical format.
    """

    source = order_points(points)

    destination = np.array(
        [
            [0, 0],
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
    # 1. Load original photograph
    # ---------------------------------------------

    image = cv2.imread(IMAGE_PATH)

    if image is None:
        raise FileNotFoundError(f"Could not load: {IMAGE_PATH}")

    original_height, original_width = image.shape[:2]

    print()
    print("=" * 60)
    print("INPUT")
    print("=" * 60)

    print(f"Image: {IMAGE_PATH}")

    print(f"Size: {original_width}x{original_height}")

    # ---------------------------------------------
    # 2. Add padding
    # ---------------------------------------------
    #
    # DocAligner's own demo uses padding to help
    # detect corners that are very close to or
    # slightly outside the original image boundary.
    # ---------------------------------------------

    padded = cv2.copyMakeBorder(
        image,
        PADDING,
        PADDING,
        PADDING,
        PADDING,
        borderType=cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )

    padded_path = os.path.join(
        OUTPUT_DIR,
        "01_padded_input.jpg",
    )

    cv2.imwrite(
        padded_path,
        padded,
    )

    # ---------------------------------------------
    # 3. Initialize DocAligner
    # ---------------------------------------------
    #
    # fastvit_sa24 is currently documented as
    # the default model configuration.
    #
    # First run may download model weights.
    # ---------------------------------------------

    print()
    print("=" * 60)
    print("LOADING DOCALIGNER")
    print("=" * 60)

    model = DocAligner(
        model_cfg="fastvit_sa24",
    )

    # ---------------------------------------------
    # 4. Detect document corners
    # ---------------------------------------------
    #
    # We disable center cropping because we already
    # control the input using explicit padding.
    # ---------------------------------------------

    print()
    print("Detecting document...")

    polygon = model(
        img=padded,
        do_center_crop=False,
    )

    # ---------------------------------------------
    # 5. Validate output
    # ---------------------------------------------

    if polygon is None:
        raise RuntimeError("DocAligner returned no document polygon.")

    polygon = np.asarray(
        polygon,
        dtype=np.float32,
    )

    if polygon.size == 0:
        raise RuntimeError("DocAligner could not detect a document.")

    if polygon.shape != (4, 2):
        raise RuntimeError(
            f"Unexpected DocAligner output shape: {polygon.shape}\nExpected: (4, 2)"
        )

    # ---------------------------------------------
    # 6. Remove padding offset
    # ---------------------------------------------
    #
    # Polygon coordinates currently refer to the
    # PADDED image.
    #
    # Convert them back to the original photo's
    # coordinate system.
    # ---------------------------------------------

    polygon[:, 0] -= PADDING
    polygon[:, 1] -= PADDING

    # ---------------------------------------------
    # 7. Print detected corners
    # ---------------------------------------------

    ordered = order_points(polygon)

    print()
    print("=" * 60)
    print("DETECTED CORNERS")
    print("=" * 60)

    for label, point in zip(
        [
            "TL",
            "TR",
            "BR",
            "BL",
        ],
        ordered,
    ):
        print(f"{label}: ({point[0]:.2f}, {point[1]:.2f})")

    # ---------------------------------------------
    # 8. Save detected polygon visualization
    # ---------------------------------------------

    detection_image = draw_polygon(
        image,
        polygon,
    )

    detection_path = os.path.join(
        OUTPUT_DIR,
        "02_docaligner_detection.jpg",
    )

    cv2.imwrite(
        detection_path,
        detection_image,
    )

    # ---------------------------------------------
    # 9. Perspective warp
    # ---------------------------------------------

    canonical = warp_to_canonical(
        image,
        polygon,
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
    # 10. Done
    # ---------------------------------------------

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)

    print(f"Padded input: {padded_path}")

    print(f"Detected corners: {detection_path}")

    print(f"Canonical licence: {canonical_path}")


if __name__ == "__main__":
    main()
