import cv2
import json


IMAGE_PATH = "assets/templates/driving_license/license_template_canonical.jpg"
OUTPUT_PATH = "config/driving_license/license_rois_canonical.json"


FIELDS = [
    "surname",
    "given_names",
    "birth_place_and_date",
    "issue_date",
    "expiry_date",
    "issued_place",
    "license_number",
    "address",
    "categories",
    "serial_number",
]


image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(f"Could not load image: {IMAGE_PATH}")


# Always work with one fixed licence size.
canonical = cv2.imread(IMAGE_PATH)

img_height, img_width, _ = image.shape


rois = {}


for field_name in FIELDS:
    print()
    print("=" * 50)
    print(f"Select ROI for: {field_name}")
    print("Drag a rectangle around the VALUE area.")
    print("Press ENTER or SPACE when done.")
    print("Press C to cancel.")
    print("=" * 50)

    x, y, width, height = cv2.selectROI(
        f"Select: {field_name}",
        image,
        showCrosshair=True,
        fromCenter=False,
    )

    cv2.destroyWindow(f"Select: {field_name}")

    if width == 0 or height == 0:
        print(f"Skipped: {field_name}")
        continue

    x1 = x
    y1 = y
    x2 = x + width
    y2 = y + height

    # Store normalized coordinates.
    # This makes the ROI independent of image resolution.
    rois[field_name] = {
        "x1": x1 / img_width,
        "y1": y1 / img_height,
        "x2": x2 / img_width,
        "y2": y2 / img_height,
    }

    # Show selected ROI permanently on the image.
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        2,
    )

    cv2.putText(
        image,
        field_name,
        (x1, max(y1 - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
    )


with open(
    OUTPUT_PATH,
    "w",
    encoding="utf-8",
) as file:
    json.dump(
        rois,
        file,
        indent=4,
    )


cv2.imwrite(
    "outputs/tools/roi/license_rois_preview.jpg",
    image,
)


print()
print("Done.")
print(f"ROIs saved to: {OUTPUT_PATH}")
print("Preview saved to: license_rois_preview.jpg")
