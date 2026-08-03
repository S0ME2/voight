import json

import cv2


# =========================================================
# CONFIG
# =========================================================

IMAGE_PATH = "assets/samples/driving_license/test_license_data_crop.jpg"

OUTPUT_JSON = "config/driving_license/field_rois_crop.json"
OUTPUT_PREVIEW = "outputs/tools/roi/field_rois_crop_preview.jpg"


FIELDS = [
    "surname",
    "given_names",
    "birth_place_and_date",
    "issue_date",
    "expiry_date",
    "issued_place",
    "personal_id",
    "license_number",
    "address",
    "categories",
    "serial_number",
]


# =========================================================
# LOAD IMAGE
# =========================================================

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")


image_height, image_width = image.shape[:2]


print()
print("=" * 60)
print("CREATE FIELD ROIs")
print("=" * 60)

print(f"Image: {IMAGE_PATH}")

print(f"Size: {image_width}x{image_height}")

print()
print("For each field:")
print("  1. Draw a generous rectangle around the field.")
print("  2. Include the field number if it is nearby.")
print("  3. Include all lines belonging to the field.")
print("  4. Avoid overlapping neighbouring fields too much.")
print("  5. Press ENTER or SPACE to confirm.")
print()
print("Press C inside the ROI window to cancel selection.")
print()


# =========================================================
# CREATE ROIs
# =========================================================

rois = {}

# Copy used for displaying already-created ROIs.
preview = image.copy()


for field_name in FIELDS:
    print()
    print("=" * 60)
    print(f"SELECT ROI: {field_name}")
    print("=" * 60)

    if field_name == "given_names":
        print("Include ALL name lines, for example:\n2. HUSNIDDIN\n   MIRZOHID O'G'LI")

    elif field_name == "birth_place_and_date":
        print("Include the complete field 3 row:\n3. TOSHLOQ TUMANI 19.10.2005")

    elif field_name == "address":
        print("Include ALL address lines.")

    elif field_name == "issue_date":
        print(
            "Include only the 4a / issue-date area.\nDo not include the 4b expiry date."
        )

    elif field_name == "expiry_date":
        print("Include only the 4b / expiry-date area.")

    print()

    window_name = f"Select ROI: {field_name}"

    x, y, width, height = cv2.selectROI(
        window_name,
        preview,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow(window_name)

    # ---------------------------------------------
    # Handle skipped ROI
    # ---------------------------------------------

    if width == 0 or height == 0:
        print(f"Skipped: {field_name}")
        continue

    # ---------------------------------------------
    # Pixel coordinates
    # ---------------------------------------------

    x1 = int(x)
    y1 = int(y)

    x2 = int(x + width)

    y2 = int(y + height)

    # ---------------------------------------------
    # Save NORMALIZED coordinates
    # ---------------------------------------------
    #
    # Example:
    #
    # x1 = 0.05
    # y1 = 0.10
    #
    # rather than:
    #
    # x1 = 25 pixels
    # y1 = 40 pixels
    #
    # This allows the same ROI configuration to
    # work if the cropped image is later scaled.
    # ---------------------------------------------

    rois[field_name] = {
        "x1": x1 / image_width,
        "y1": y1 / image_height,
        "x2": x2 / image_width,
        "y2": y2 / image_height,
    }

    # ---------------------------------------------
    # Draw selected ROI on preview
    # ---------------------------------------------

    cv2.rectangle(
        preview,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    cv2.putText(
        preview,
        field_name,
        (
            x1,
            max(
                y1 - 5,
                15,
            ),
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
        cv2.LINE_AA,
    )

    print(f"Saved ROI for {field_name}: ({x1}, {y1}) -> ({x2}, {y2})")


# =========================================================
# SAVE JSON
# =========================================================

with open(
    OUTPUT_JSON,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        rois,
        file,
        ensure_ascii=False,
        indent=4,
    )


# =========================================================
# SAVE PREVIEW
# =========================================================

success = cv2.imwrite(
    OUTPUT_PREVIEW,
    preview,
)

if not success:
    raise RuntimeError(f"Could not save preview: {OUTPUT_PREVIEW}")


# =========================================================
# RESULT
# =========================================================

print()
print("=" * 60)
print("DONE")
print("=" * 60)

print(f"ROIs saved to: {OUTPUT_JSON}")

print(f"Preview saved to: {OUTPUT_PREVIEW}")

print()
print(f"Created {len(rois)} of {len(FIELDS)} ROIs.")
