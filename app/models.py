import time
from collections.abc import Mapping
from typing import Any, Callable

from app.config import Settings, TextModelSettings

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
TEXT_DETECTION_MODEL_NAME = OCR_MODEL_CONFIG["text_detection_model_name"]
TEXT_RECOGNITION_MODEL_NAME = OCR_MODEL_CONFIG["text_recognition_model_name"]


class Models:
    """The only place heavy runtime models are created and retained."""

    def __init__(
        self,
        settings: Settings,
        *,
        text_recognizer_factories: Mapping[
            str, Callable[[TextModelSettings], Any]
        ] | None = None,
    ):
        self.settings = settings
        self.text_recognizer_factories = dict(text_recognizer_factories or {})
        self._ocr: Any | None = None
        self._text_detector: Any | None = None
        self._text_recognizer: Any | None = None
        self._process_text_recognizer: Any | None = None
        self._profile_batch_runner: Any | None = None
        self._mrz_scanner: Any | None = None
        self._mrz_recognition_scanner: Any | None = None
        self._mrz_recognizer: Any | None = None
        self._document_aligner: Any | None = None
        self._mrz_localizer: Any | None = None
        self._document_localizer: Any | None = None
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

            return PaddleOCR(**OCR_MODEL_CONFIG, device=self._paddle_device())

        return self._get_or_load("_ocr", "ocr", load)

    def mrz_scanner(self) -> Any:
        def load() -> Any:
            from mrzscanner import MRZScanner, ModelType

            return MRZScanner(
                model_type=ModelType.detection,
                detection_cfg=self.settings.mrz.scanner_config,
                backend=self._onnx_backend(),
                gpu_id=self.settings.runtime.gpu_id,
                session_option=self._onnx_session_options(),
            )

        return self._get_or_load("_mrz_scanner", "mrz_scanner", load)

    def text_detector(self) -> Any:
        def load() -> Any:
            from paddleocr import TextDetection
            from app.inference.paddle import PaddleTextDetector

            selection = self.settings.models.text_detector
            if selection.backend != "paddle":
                raise ValueError(f"unknown text detector backend: {selection.backend}")
            options = dict(
                model_name=selection.model,
                model_dir=self._paddle_model_dir(selection.model),
                device=self._paddle_device(),
                thresh=OCR_PREDICT_CONFIG["text_det_thresh"],
                box_thresh=OCR_PREDICT_CONFIG["text_det_box_thresh"],
                unclip_ratio=OCR_PREDICT_CONFIG["text_det_unclip_ratio"],
            )
            return PaddleTextDetector(TextDetection(**options))

        return self._get_or_load("_text_detector", "text_detector", load)

    def text_recognizer(self) -> Any:
        return self._get_or_load(
            "_text_recognizer", "text_recognizer", self._load_text_recognizer
        )

    def _load_text_recognizer(self) -> Any:
        selection = self.settings.models.text_recognizer
        if factory := self.text_recognizer_factories.get(selection.backend):
            return factory(selection)
        if selection.backend != "paddle":
            raise ValueError(f"unknown text recognizer backend: {selection.backend}")
        from paddleocr import TextRecognition
        from app.inference.paddle import PaddleTextRecognizer

        runtime = self.settings.runtime
        options = dict(
            model_name=selection.model,
            model_dir=self._paddle_model_dir(selection.model),
            device=self._paddle_device(),
            cpu_threads=runtime.cpu_threads,
        )
        if runtime.text_recognition_enable_hpi or runtime.text_recognition_use_tensorrt or runtime.text_recognition_precision != "fp32":
            options.update(
                enable_hpi=runtime.text_recognition_enable_hpi,
                use_tensorrt=runtime.text_recognition_use_tensorrt,
                precision=runtime.text_recognition_precision,
            )
        return PaddleTextRecognizer(TextRecognition(**options))

    def process_text_recognizer(self) -> Any:
        if self._process_text_recognizer is None:
            from app.inference.paddle import ProcessTextRecognizer as ProcessTextRecognizerAdapter
            from app.inference.recognition_workers import ProcessTextRecognizer

            runtime = self.settings.runtime
            selection = self.settings.models.text_recognizer
            if selection.backend != "paddle":
                raise ValueError("TEXT_RECOGNITION_PROCESSES requires the paddle backend")
            self._process_text_recognizer = ProcessTextRecognizerAdapter(
                ProcessTextRecognizer(
                    model_dir=self._paddle_model_dir(selection.model),
                    processes=runtime.text_recognition_processes,
                    cpu_threads=runtime.cpu_threads,
                )
            )
        return self._process_text_recognizer

    def profile_batch_runner(self) -> Any:
        """Return the one bounded inference coordinator owned by this process."""
        if self._profile_batch_runner is None:
            from app.inference.batch import BatchedOcr, ProfileBatchRunner
            from app.inference.packing import recognition_batch_packer

            runtime = self.settings.runtime
            self._profile_batch_runner = ProfileBatchRunner(
                BatchedOcr(
                    self.text_detector(),
                    self.process_text_recognizer()
                    if runtime.text_recognition_processes > 1
                    else self.text_recognizer(),
                    detection_batch_size=runtime.text_detection_batch_size,
                    recognition_batch_size=runtime.text_recognition_batch_size,
                    recognition_packer=recognition_batch_packer(runtime.text_recognition_packing),
                ),
                {
                    "docaligner": self.document_localizer(),
                    "mrz": self.mrz_localizer(),
                },
                self.settings.mrz,
                localization_batch_size=runtime.localization_batch_size,
                mrz_recognizer=self.mrz_recognizer(),
                mrz_recognition_batch_size=runtime.mrz_recognition_batch_size,
                max_items=self.settings.batch.max_files * 2,
            )
        return self._profile_batch_runner

    def document_aligner(self) -> Any:
        def load() -> Any:
            from docaligner import DocAligner

            return DocAligner(
                model_cfg=self.settings.driving_license.aligner_model,
                backend=self._onnx_backend(),
                gpu_id=self.settings.runtime.gpu_id,
                session_option=self._onnx_session_options(),
            )

        return self._get_or_load("_document_aligner", "document_aligner", load)

    def mrz_localizer(self) -> Any:
        def load() -> Any:
            from app.inference.localization import MrzScannerBatchLocalizer

            backend = self.settings.models.localization.mrz_backend
            if backend != "mrzscanner":
                raise ValueError(f"unknown MRZ localizer backend: {backend}")
            return MrzScannerBatchLocalizer(self.mrz_scanner())

        return self._get_or_load("_mrz_localizer", "mrz_localizer", load)

    def mrz_recognizer(self) -> Any | None:
        backend = self.settings.models.mrz.recognizer_backend
        if backend == "generic-paddle":
            return None
        if backend != "mrzscanner":
            raise ValueError(f"unknown MRZ recognizer backend: {backend}")

        def load() -> Any:
            from mrzscanner import MRZScanner, ModelType
            from app.inference.mrzscanner import MrzScannerRecognizer

            scanner = MRZScanner(
                model_type=ModelType.recognition,
                recognition_cfg=self.settings.models.mrz.recognizer_model,
                backend=self._onnx_backend(),
                gpu_id=self.settings.runtime.gpu_id,
                session_option=self._onnx_session_options(),
            )
            self._mrz_recognition_scanner = scanner
            return MrzScannerRecognizer(scanner)

        return self._get_or_load("_mrz_recognizer", "mrz_recognizer", load)

    def document_localizer(self) -> Any:
        def load() -> Any:
            from app.inference.localization import DocAlignerBatchLocalizer

            backend = self.settings.models.localization.document_backend
            if backend != "docaligner":
                raise ValueError(f"unknown document localizer backend: {backend}")
            return DocAlignerBatchLocalizer(self.document_aligner())

        return self._get_or_load("_document_localizer", "document_localizer", load)

    def _paddle_device(self) -> str:
        runtime = self.settings.runtime
        return "cpu" if runtime.target == "cpu" else f"gpu:{runtime.gpu_id}"

    def _paddle_model_dir(self, name: str) -> str | None:
        root = self.settings.models.directory
        return None if root is None else str(root / "official_models" / name)

    def _onnx_backend(self) -> Any:
        from capybara import Backend

        return Backend.cpu if self.settings.runtime.target == "cpu" else Backend.cuda

    def _onnx_session_options(self) -> dict[str, int]:
        return {"intra_op_num_threads": self.settings.runtime.cpu_threads}

    def is_loaded(self, name: str) -> bool:
        attributes = {
            "ocr": "_ocr",
            "text_detector": "_text_detector",
            "text_recognizer": "_text_recognizer",
            "mrz_scanner": "_mrz_scanner",
            "document_aligner": "_document_aligner",
            "mrz_localizer": "_mrz_localizer",
            "mrz_recognizer": "_mrz_recognizer",
            "document_localizer": "_document_localizer",
        }
        try:
            return getattr(self, attributes[name]) is not None
        except KeyError as exc:
            raise ValueError(f"Unknown model: {name}") from exc

    def recorded_load_seconds(self, name: str) -> float | None:
        return self._load_seconds.get(name)

    def preload(self) -> None:
        if self.settings.preload:
            self.profile_batch_runner()
            if self.settings.runtime.text_recognition_processes > 1:
                self.process_text_recognizer().start()

    def close(self) -> None:
        if self._process_text_recognizer is not None:
            self._process_text_recognizer.close()

    def readiness(self) -> dict[str, Any]:
        from app.inference.backends import validate_runtime

        runner = self.profile_batch_runner()
        return validate_runtime(
            self.settings,
            localizers=tuple(runner.localizers.values()),
        )
