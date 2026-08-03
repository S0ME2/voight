from typing import Any

import numpy as np

from app.artifacts import ArtifactWriter
from app.models import OCR_PREDICT_CONFIG


def recognize(image: np.ndarray, model: Any, artifacts: ArtifactWriter) -> list[dict[str, Any]]:
    results = list(model.predict(image, **OCR_PREDICT_CONFIG))
    for index, result in enumerate(results):
        artifacts.save_model_image(f"06_paddle_ocr_annotated_{index}.jpg", result)
        artifacts.save_model_json(f"07_paddle_ocr_result_{index}.json", result)
    return tokens_from_results(results)


def tokens_from_results(results: list[Any]) -> list[dict[str, Any]]:
    tokens = []
    for result in results:
        for text, score, box in zip(result["rec_texts"], result["rec_scores"], result["rec_boxes"]):
            x1, y1, x2, y2 = map(float, np.asarray(box).reshape(-1)[:4])
            tokens.append({"index": len(tokens), "text": str(text).strip(), "score": float(score), "x1": x1, "y1": y1, "x2": x2, "y2": y2, "center_x": (x1 + x2) / 2, "center_y": (y1 + y2) / 2, "height": max(1.0, y2 - y1)})
    return tokens
