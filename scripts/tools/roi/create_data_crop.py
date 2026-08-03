import cv2
import json

folder = "outputs/tools/roi"

IMAGE_PATH = "assets/samples/passport/passport_crop.png"

OUTPUT_JSON = "config/passport/data_crop.json"
OUTPUT_PREVIEW = f"./{folder}/data_crop_preview.jpg"
OUTPUT_CROPPED = "assets/samples/passport/test_passport_data_crop.png"


# =========================================================
# LOAD IMAGE
# =========================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")


image_height, image_width = image.shape[:2]


# =========================================================
# SELECT USEFUL DATA REGION
# =========================================================

print()
print("=" * 60)
print("SELECT USEFUL DATA REGION")
print("=" * 60)
print()
print("Select one rectangle containing all useful fields.")
print("Exclude:")
print("  - portrait")
print("  - header")
print("  - right-side region names")
print("  - signature")
print("  - other unnecessary content")
print()
print("Press ENTER or SPACE when done.")
print()


x, y, width, height = cv2.selectROI(
    "Select useful data region",
    image,
    showCrosshair=True,
    fromCenter=False,
)

cv2.destroyAllWindows()


if width == 0 or height == 0:
    raise RuntimeError("No ROI was selected.")


# =========================================================
# CALCULATE COORDINATES
# =========================================================

x1 = x
y1 = y

x2 = x + width
y2 = y + height


# =========================================================
# SAVE NORMALIZED ROI
# =========================================================

data = {
    "data_crop": {
        "x1": x1 / image_width,
        "y1": y1 / image_height,
        "x2": x2 / image_width,
        "y2": y2 / image_height,
    }
}


with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        data,
        file,
        indent=4,
    )


# =========================================================
# SAVE CROPPED IMAGE
# =========================================================

cropped = image[
    y1:y2,
    x1:x2,
]


success = cv2.imwrite(
    OUTPUT_CROPPED,
    cropped,
)

if not success:
    raise RuntimeError(f"Could not save: {OUTPUT_CROPPED}")


# =========================================================
# SAVE PREVIEW
# =========================================================

preview = image.copy()

cv2.rectangle(
    preview,
    (x1, y1),
    (x2, y2),
    (0, 255, 0),
    3,
)

cv2.putText(
    preview,
    "USEFUL DATA CROP",
    (
        x1,
        max(y1 - 10, 20),
    ),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2,
    cv2.LINE_AA,
)


success = cv2.imwrite(
    OUTPUT_PREVIEW,
    preview,
)

if not success:
    raise RuntimeError(f"Could not save: {OUTPUT_PREVIEW}")


# =========================================================
# RESULT
# =========================================================

print()
print("=" * 60)
print("DONE")
print("=" * 60)

print(f"Crop configuration: {OUTPUT_JSON}")

print(f"Crop preview:       {OUTPUT_PREVIEW}")

print(f"Cropped image:      {OUTPUT_CROPPED}")

print()
print(f"Original size: {image_width}x{image_height}")

print(f"Cropped size:  {width}x{height}")
