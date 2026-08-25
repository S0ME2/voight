import os

from paddleocr import TextImageUnwarping


# =========================================================
# CONFIG
# =========================================================

IMAGE_PATH = "assets/samples/driving_license/test_license.jpg"

OUTPUT_DIR = "outputs/experiments/alignment/uvdoc"


# =========================================================
# MAIN
# =========================================================


def main() -> None:
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    print()
    print("=" * 60)
    print("UVDOC DOCUMENT RECTIFICATION TEST")
    print("=" * 60)

    print(f"Input image: {IMAGE_PATH}")

    print()

    # -----------------------------------------------------
    # Initialize PaddleOCR's UVDoc model
    # -----------------------------------------------------
    #
    # The first run may download the pretrained model.
    #
    # By default PaddleOCR uses its default local inference
    # engine and automatically chooses the available device.
    # -----------------------------------------------------

    model = TextImageUnwarping(
        model_name="UVDoc",
    )

    # -----------------------------------------------------
    # Run document rectification
    # -----------------------------------------------------

    results = model.predict(
        IMAGE_PATH,
        batch_size=1,
    )

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    for index, result in enumerate(results):
        print()
        print(f"Processing result #{index}")

        # Print PaddleOCR result information.
        result.print()

        # Save the rectified image.
        #
        # When save_path is a directory, PaddleOCR creates
        # the output image inside that directory.
        result.save_to_img(
            save_path=OUTPUT_DIR,
        )

        # Save metadata/result JSON as well.
        json_path = os.path.join(
            OUTPUT_DIR,
            f"uvdoc_result_{index}.json",
        )

        result.save_to_json(
            save_path=json_path,
        )

        print()
        print(f"Rectified image saved in: {OUTPUT_DIR}")

        print(f"Result JSON: {json_path}")

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
