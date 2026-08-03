import time
from typing import Any, Callable

from app.config import Settings

OCR_MODEL_CONFIG = {
    "text_detection_model_name": "PP-OCRv6_medium_det",
    "text_recognition_model_name": "PP-OCRv6_medium_rec",
    "use_doc_orientation_classify": False,
    "use_doc_unwarping": False,
    "use_textline_orientation": False,
}
OCR_PREDICT_CONFIG = {
    "text_det_thresh": 0.30,
    "text_det_box_thresh": 0.50,
    "text_det_unclip_ratio": 2.00,
    "text_rec_score_thresh": 0.0,
}


class Models:
    """The only place heavy runtime models are created and retained."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._ocr: Any | None = None
        self._mrz_scanner: Any | None = None
        self._document_aligner: Any | None = None
        self._load_seconds: dict[str, float] = {}

    def _get_or_load(
        self,
        attribute: str,
        name: str,
        loader: Callable[[], Any],
    ) -> Any:
        current = getattr(self, attribute)
        if current is None:
            started = time.perf_counter()
            current = loader()
            self._load_seconds[name] = time.perf_counter() - started
            setattr(self, attribute, current)
        return current

    def ocr(self) -> Any:
        def load() -> Any:
            from paddleocr import PaddleOCR

            return PaddleOCR(**OCR_MODEL_CONFIG, device=self.settings.ocr.device)

        return self._get_or_load("_ocr", "ocr", load)

    def mrz_scanner(self) -> Any:
        def load() -> Any:
            from mrzscanner import MRZScanner, ModelType

            return MRZScanner(
                model_type=ModelType.detection,
                detection_cfg=self.settings.mrz.scanner_config,
            )

        return self._get_or_load("_mrz_scanner", "mrz_scanner", load)

    def document_aligner(self) -> Any:
        def load() -> Any:
            from docaligner import DocAligner

            return DocAligner(
                model_cfg=self.settings.driving_license.aligner_model
            )

        return self._get_or_load("_document_aligner", "document_aligner", load)

    def is_loaded(self, name: str) -> bool:
        attributes = {
            "ocr": "_ocr",
            "mrz_scanner": "_mrz_scanner",
            "document_aligner": "_document_aligner",
        }
        try:
            return getattr(self, attributes[name]) is not None
        except KeyError as exc:
            raise ValueError(f"Unknown model: {name}") from exc

    def recorded_load_seconds(self, name: str) -> float | None:
        return self._load_seconds.get(name)

    def preload(self) -> None:
        if self.settings.preload:
            self.ocr()
            self.mrz_scanner()
            self.document_aligner()
