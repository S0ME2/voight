import cv2
import numpy as np

name = "test_license"

IMAGE_PATH = f"assets/samples/driving_license/{name}.jpg"
OUTPUT_PATH = f"outputs/tools/alignment/{name}_canonical.jpg"

CANONICAL_WIDTH = 1000
CANONICAL_HEIGHT = 630


points = []


def order_points(points_array):
    """
    Convert four arbitrary corner points into:

    top-left
    top-right
    bottom-right
    bottom-left
    """

    ordered = np.zeros(
        (4, 2),
        dtype=np.float32,
    )

    point_sum = points_array.sum(axis=1)
    point_diff = np.diff(
        points_array,
        axis=1,
    ).reshape(-1)

    # Smallest x+y
    ordered[0] = points_array[np.argmin(point_sum)]

    # Largest x+y
    ordered[2] = points_array[np.argmax(point_sum)]

    # Smallest y-x
    ordered[1] = points_array[np.argmin(point_diff)]

    # Largest y-x
    ordered[3] = points_array[np.argmax(point_diff)]

    return ordered


def mouse_callback(
    event,
    x,
    y,
    flags,
    param,
):
    if event == cv2.EVENT_LBUTTONDOWN and len(points) < 4:
        points.append((x, y))

        print(f"Point {len(points)}: ({x}, {y})")


image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Could not load: {IMAGE_PATH}")


window_name = "Select 4 licence corners"

cv2.namedWindow(window_name)

cv2.setMouseCallback(
    window_name,
    mouse_callback,
)


while True:
    preview = image.copy()

    for index, point in enumerate(points):
        cv2.circle(
            preview,
            point,
            6,
            (0, 255, 0),
            -1,
        )

        cv2.putText(
            preview,
            str(index + 1),
            (
                point[0] + 8,
                point[1] - 8,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    cv2.imshow(
        window_name,
        preview,
    )

    key = cv2.waitKey(20) & 0xFF

    # ESC
    if key == 27:
        break

    # R = reset points
    if key == ord("r"):
        points.clear()

    # ENTER after selecting 4 corners
    if key in {10, 13} and len(points) == 4:
        break


cv2.destroyAllWindows()


if len(points) != 4:
    raise RuntimeError("You must select exactly 4 corners.")


source_points = order_points(
    np.array(
        points,
        dtype=np.float32,
    )
)


destination_points = np.array(
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
    source_points,
    destination_points,
)


canonical = cv2.warpPerspective(
    image,
    transform,
    (
        CANONICAL_WIDTH,
        CANONICAL_HEIGHT,
    ),
)


cv2.imwrite(
    OUTPUT_PATH,
    canonical,
)


print(f"Saved canonical document to: {OUTPUT_PATH}")
