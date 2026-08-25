import json
import os

from paddleocr import PaddleOCR


IMAGE_PATH = "assets/samples/driving_license/test_license_data_crop.jpg"
OUTPUT_DIR = "outputs/experiments/roi/data_crop"


def main():
    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    ocr = PaddleOCR(
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    results = ocr.predict(IMAGE_PATH)

    for index, result in enumerate(results):
        # -----------------------------------------
        # Save PaddleOCR annotated image
        # -----------------------------------------

        if index == 0:
            annotated_path = os.path.join(
                OUTPUT_DIR,
                "paddle_ocr_annotated.jpg",
            )
        else:
            annotated_path = os.path.join(
                OUTPUT_DIR,
                f"paddle_ocr_annotated_{index}.jpg",
            )

        result.save_to_img(annotated_path)

        # -----------------------------------------
        # Extract raw OCR information
        # -----------------------------------------

        texts = result["rec_texts"]
        scores = result["rec_scores"]
        boxes = result["rec_boxes"]

        raw_results = []

        for text, score, box in zip(
            texts,
            scores,
            boxes,
        ):
            raw_results.append(
                {
                    "text": text,
                    "score": float(score),
                    "box": [int(value) for value in box],
                }
            )

        # -----------------------------------------
        # Save raw OCR JSON
        # -----------------------------------------

        if index == 0:
            json_path = os.path.join(
                OUTPUT_DIR,
                "raw_ocr.json",
            )
        else:
            json_path = os.path.join(
                OUTPUT_DIR,
                f"raw_ocr_{index}.json",
            )

        with open(
            json_path,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                raw_results,
                file,
                ensure_ascii=False,
                indent=4,
            )

        # -----------------------------------------
        # Print OCR results
        # -----------------------------------------

        print()
        print("=" * 80)
        print("OCR RESULTS")
        print("=" * 80)

        for item in raw_results:
            print(f"{item['text']!r} | score={item['score']:.3f} | box={item['box']}")

        print()
        print("=" * 80)
        print("SAVED")
        print("=" * 80)

        print(f"Annotated OCR: {annotated_path}")

        print(f"Raw OCR JSON: {json_path}")


if __name__ == "__main__":
    main()
