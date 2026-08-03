import os
import re

import cv2
from paddleocr import PaddleOCR


MRZ_PATTERN = re.compile(r"^[A-Z0-9<]{30,50}$")


def looks_like_mrz(text: str) -> bool:
    return bool(MRZ_PATTERN.fullmatch(text))


def preprocess_image(image_path: str):
    """Apply the winning grayscale preprocessing."""
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def run_ocr(ocr: PaddleOCR, image_path: str, output_dir: str) -> None:
    # Winning preprocessing: grayscale.
    image = preprocess_image(image_path)

    # Winning detection configuration.
    results = ocr.predict(
        image,
        text_det_thresh=0.30,
        text_det_box_thresh=0.50,
        text_det_unclip_ratio=2.00,
        text_rec_score_thresh=0.0,
    )

    filename_with_ext = os.path.basename(image_path)
    filename = os.path.splitext(filename_with_ext)[0]

    for result in results:
        print(f"\n{'=' * 50} OCR RESULT({filename_with_ext}) {'=' * 50}")

        texts = result["rec_texts"]
        scores = result["rec_scores"]

        for text, score in zip(texts, scores):
            if looks_like_mrz(text):
                print(f"{text}: {score:.2f}")

        image_output = os.path.join(output_dir, f"annotated_{filename}.png")
        result.save_to_img(image_output)

        json_output = os.path.join(output_dir, f"result_{filename}.json")
        result.save_to_json(json_output)


if __name__ == "__main__":
    input_dir = "./dataset/source/"
    output_dir = "./outputs/ocr/best_run/"

    os.makedirs(output_dir, exist_ok=True)

    # Winning model configuration.
    ocr = PaddleOCR(
        text_detection_model_name="PP-OCRv6_medium_det",
        text_recognition_model_name="PP-OCRv6_medium_rec",
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=False,
    )

    valid_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    for filename in os.listdir(input_dir):
        image_path = os.path.join(input_dir, filename)

        if not os.path.isfile(image_path):
            continue

        if os.path.splitext(filename)[1].lower() not in valid_extensions:
            continue

        run_ocr(ocr=ocr, image_path=image_path, output_dir=output_dir)
