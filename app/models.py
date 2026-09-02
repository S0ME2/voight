import os
import time
from collections.abc import Mapping
from typing import Any, Callable

from app.config import Settings, TextModelSettings

OCR_PREDICT_CONFIG = {
    "text_det_thresh": 0.30,
    "text_det_box_thresh": 0.50,
    "text_det_unclip_ratio": 2.00,
    "text_rec_score_thresh": 0.0,
}
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
        self._text_detector: Any | None = None
        self._text_recognizer: Any | None = None
        self._process_text_recognizer: Any | None = None
        self._profile_batch_runner: Any | None = None
        self._verification_ocr: Any | None = None
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
            trace_dir = os.getenv("VOIGHT_BENCHMARK_TRACE_DIR")
            if trace_dir:
                try:
                    import json
                    from pathlib import Path

                    target = Path(trace_dir)
                    target.mkdir(parents=True, exist_ok=True)
                    (target / f"model-load-{os.getpid()}-{name}-{time.time_ns()}.json").write_text(
                        json.dumps({"model": name, "seconds": self._load_seconds[name]}),
                        encoding="utf-8",
                    )
                except OSError:
                    pass
        return current

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
                cpu_threads=self.settings.runtime.cpu_threads,
                thresh=OCR_PREDICT_CONFIG["text_det_thresh"],
                box_thresh=OCR_PREDICT_CONFIG["text_det_box_thresh"],
                unclip_ratio=OCR_PREDICT_CONFIG["text_det_unclip_ratio"],
            )
            return PaddleTextDetector(TextDetection(**options), self.settings.runtime)

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
                    model_name=selection.model,
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
                    mrz_recognition_batch_size=runtime.mrz_recognition_batch_size,
                    recognition_packer=recognition_batch_packer(runtime.text_recognition_packing),
                    mrz_contrast=self.settings.mrz.contrast,
                    detector_preprocessing=runtime.text_detector_preprocessing,
                    visible_preprocessing=runtime.visible_recognition_preprocessing,
                    mrz_preprocessing=runtime.mrz_preprocessing,
                ),
                {
                    "docaligner": self.document_localizer(),
                    "mrz": self.mrz_localizer(),
                },
                self.settings.mrz,
                localization_batch_size=runtime.localization_batch_size,
                mrz_recognizer=self.mrz_recognizer(),
                mrz_recognition_batch_size=runtime.mrz_recognition_batch_size,
                mrz_recognizer_config=self.settings.models.mrz.recognizer_model,
                max_items=self.settings.batch.max_files * 2,
            )
        return self._profile_batch_runner

    def verification_ocr(self) -> Any:
        """Return whole-image OCR while sharing the process-owned text models."""
        if self._verification_ocr is None:
            from app.inference.batch import BatchedOcr
            from app.inference.packing import recognition_batch_packer

            runtime = self.settings.runtime
            verification = self.settings.verification
            if verification is None:
                from app.config import VerificationBatchSettings

                verification = VerificationBatchSettings(
                    runtime.text_detection_batch_size,
                    runtime.text_recognition_batch_size,
                )
            self._verification_ocr = BatchedOcr(
                self.text_detector(),
                self.process_text_recognizer() if runtime.text_recognition_processes > 1 else self.text_recognizer(),
                detection_batch_size=verification.text_detection_batch_size,
                recognition_batch_size=verification.text_recognition_batch_size,
                mrz_recognition_batch_size=runtime.mrz_recognition_batch_size,
                recognition_packer=recognition_batch_packer(runtime.text_recognition_packing),
                detector_preprocessing=runtime.text_detector_preprocessing,
                visible_preprocessing=runtime.visible_recognition_preprocessing,
            )
        return self._verification_ocr

    def document_aligner(self) -> Any:
        def load() -> Any:
            import docaligner

            model_type = self.settings.driving_license.aligner_model_type
            types = getattr(docaligner, "ModelType", None)
            return docaligner.DocAligner(
                model_type=getattr(types, model_type) if types else model_type,
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
        if backend not in {"mrzscanner", "mrzscanner-spotting"}:
            raise ValueError(f"unknown MRZ recognizer backend: {backend}")

        def load() -> Any:
            from mrzscanner import MRZScanner, ModelType
            from app.inference.mrzscanner import MrzScannerRecognizer, MrzScannerSpottingRecognizer

            if backend == "mrzscanner-spotting":
                scanner = MRZScanner(
                    model_type=ModelType.spotting,
                    spotting_cfg=self.settings.models.mrz.recognizer_model,
                    backend=self._onnx_backend(),
                    gpu_id=self.settings.runtime.gpu_id,
                    session_option=self._onnx_session_options(),
                )
                self._mrz_recognition_scanner = scanner
                return MrzScannerSpottingRecognizer(scanner)
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
            from app.inference.localization import DocAlignerBatchLocalizer, PointDocAlignerLocalizer

            backend = self.settings.models.localization.document_backend
            if backend != "docaligner":
                raise ValueError(f"unknown document localizer backend: {backend}")
            aligner = self.document_aligner()
            return PointDocAlignerLocalizer(aligner) if self.settings.driving_license.aligner_model_type == "point" else DocAlignerBatchLocalizer(aligner)

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

    def configuration(self) -> dict[str, Any]:
        mrz_backend = self.settings.models.mrz.recognizer_backend
        generic_mrz = mrz_backend == "generic-paddle"
        detector_resize = self._text_detector.resize_configuration() if self._text_detector is not None else None
        if detector_resize is None:
            detector_resize = {
                "effective_percent": round(self.settings.runtime.text_detector_pixel_scale ** 2 * 100, 2),
                "pixel_scale": self.settings.runtime.text_detector_pixel_scale,
                "limit_side_len_override": self.settings.runtime.text_detector_limit_side_len,
                "preprocessing": "original",
                "loaded": False,
            }
        try:
            import cv2

            opencv_threads = cv2.getNumThreads()
        except ImportError:
            opencv_threads = None
        detector_threads = self.settings.runtime.cpu_threads
        recognizer_threads = self.settings.runtime.cpu_threads
        if self._text_detector is not None:
            detector_threads = getattr(self._text_detector.model, "_common_args", {}).get("cpu_threads", detector_threads)
        if self._text_recognizer is not None:
            recognizer_threads = getattr(self._text_recognizer.model, "_common_args", {}).get("cpu_threads", recognizer_threads)
        return {
            "runtime": {
                "target": self.settings.runtime.target,
                "paddle_cpu_threads": self.settings.runtime.cpu_threads,
                "onnx_intra_op_threads": self.settings.runtime.cpu_threads,
                "onnx_inter_op_threads": 0,
                "omp_num_threads": int(os.getenv("OMP_NUM_THREADS", "1")),
                "opencv_threads": opencv_threads,
            },
            "batch": {
                "localization": self.settings.runtime.localization_batch_size,
                "detection": self.settings.runtime.text_detection_batch_size,
                "recognition": self.settings.runtime.text_recognition_batch_size,
                "mrz_recognition": self.settings.runtime.mrz_recognition_batch_size,
            },
            "verification_batch": {
                "detection": (self.settings.verification or self.settings.runtime).text_detection_batch_size,
                "recognition": (self.settings.verification or self.settings.runtime).text_recognition_batch_size,
            },
            "recognition_packing": self.settings.runtime.text_recognition_packing,
            "recognition_acceleration": {
                "precision": self.settings.runtime.text_recognition_precision,
                "hpi": self.settings.runtime.text_recognition_enable_hpi,
                "tensorrt": self.settings.runtime.text_recognition_use_tensorrt,
            },
            "detector_preprocessing": self.settings.runtime.text_detector_preprocessing,
            "visible_recognition_preprocessing": self.settings.runtime.visible_recognition_preprocessing,
            "mrz_preprocessing": {
                "normalization": "resize/grayscale only",
                "line_contrast": self.settings.mrz.contrast,
                "variant": self.settings.runtime.mrz_preprocessing,
                "stage": "raw MRZ line crop before fixed-width packing",
            },
            "text_detector": {"backend": self.settings.models.text_detector.backend, "model": self.settings.models.text_detector.model, "path": self._paddle_model_dir(self.settings.models.text_detector.model), "cpu_threads": detector_threads, "loaded": self.is_loaded("text_detector"), "resize": detector_resize},
            "text_recognizer": {"backend": self.settings.models.text_recognizer.backend, "model": self.settings.models.text_recognizer.model, "path": self._paddle_model_dir(self.settings.models.text_recognizer.model), "cpu_threads": recognizer_threads, "loaded": self.is_loaded("text_recognizer")},
            "document_localizer": {"backend": self.settings.models.localization.document_backend, "model": self.settings.driving_license.aligner_model, "model_type": self.settings.driving_license.aligner_model_type, "model_cfg": self.settings.driving_license.aligner_model, "loaded": self.is_loaded("document_localizer")},
            "mrz_localizer": {"backend": self.settings.models.localization.mrz_backend, "model_cfg": self.settings.mrz.scanner_config, "loaded": self.is_loaded("mrz_localizer")},
            "mrz_recognizer": {
                "backend": mrz_backend,
                "model_cfg": self.settings.models.mrz.recognizer_model,
                "model_cfg_used": not generic_mrz,
                "effective_backend": self.settings.models.text_recognizer.backend if generic_mrz else mrz_backend,
                "effective_model": self.settings.models.text_recognizer.model if generic_mrz else self.settings.models.mrz.recognizer_model,
                "source": "text_recognizer" if generic_mrz else "mrzscanner",
                "loaded": self.is_loaded("mrz_recognizer") or (generic_mrz and self.is_loaded("text_recognizer")),
            },
        }
