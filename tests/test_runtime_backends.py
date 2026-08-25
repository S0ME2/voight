import unittest
import os
import sys
from types import ModuleType
from dataclasses import replace
from unittest.mock import patch

from app.config import OcrSettings, RuntimeSettings, Settings
from app.inference.backends import validate_runtime
from app.models import Models


class RuntimeBackendTests(unittest.TestCase):
    def test_gpu_readiness_checks_cuda_provider_build_and_gpu_id_with_fakes(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()
        settings = replace(
            settings,
            ocr=OcrSettings("gpu"),
            runtime=replace(settings.runtime, target="gpu", gpu_id=2),
        )

        class Cuda:
            @staticmethod
            def device_count():
                return 3

        class Device:
            cuda = Cuda()

            @staticmethod
            def is_compiled_with_cuda():
                return True

        paddle = type("Paddle", (), {"device": Device()})()
        ort = type("Ort", (), {"get_available_providers": staticmethod(lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"])})()
        localizer = type("Localizer", (), {"providers": ["CUDAExecutionProvider"]})()

        result = validate_runtime(settings, paddle_module=paddle, ort_module=ort, localizers=(localizer,))
        self.assertEqual("gpu:2", result["paddle_device"])
        self.assertEqual("CUDAExecutionProvider", result["onnx_provider"])

    def test_gpu_readiness_rejects_missing_cuda_without_importing_gpu_runtime(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()
        settings = replace(settings, ocr=OcrSettings("gpu"), runtime=replace(settings.runtime, target="gpu"))
        paddle = type("Paddle", (), {"device": type("Device", (), {"is_compiled_with_cuda": staticmethod(lambda: False)})()})()
        ort = type("Ort", (), {"get_available_providers": staticmethod(lambda: ["CPUExecutionProvider"])})()
        with self.assertRaisesRegex(RuntimeError, "CUDAExecutionProvider"):
            validate_runtime(settings, paddle_module=paddle, ort_module=ort)

    def test_model_factories_propagate_gpu_backend_id_and_paddle_device(self):
        with patch.dict(os.environ, {}, clear=True):
            base = Settings.from_env()
        settings = replace(
            base,
            ocr=OcrSettings("gpu"),
            runtime=replace(base.runtime, target="gpu", gpu_id=3, cpu_threads=7),
        )
        created = []

        class Wrapper:
            def __init__(self, **kwargs):
                created.append(kwargs)

        capybara = ModuleType("capybara")
        capybara.Backend = type("Backend", (), {"cpu": "CPU", "cuda": "CUDA"})
        mrzscanner = ModuleType("mrzscanner")
        mrzscanner.MRZScanner = Wrapper
        mrzscanner.ModelType = type("ModelType", (), {"detection": "detection"})
        docaligner = ModuleType("docaligner")
        docaligner.DocAligner = Wrapper
        paddleocr = ModuleType("paddleocr")
        paddleocr.TextDetection = Wrapper
        modules = {"capybara": capybara, "mrzscanner": mrzscanner, "docaligner": docaligner, "paddleocr": paddleocr}

        models = Models(settings)
        with patch.dict(sys.modules, modules):
            models.mrz_scanner()
            models.document_aligner()
            models.text_detector()

        self.assertEqual("CUDA", created[0]["backend"])
        self.assertEqual(3, created[0]["gpu_id"])
        self.assertEqual(7, created[0]["session_option"]["intra_op_num_threads"])
        self.assertEqual("CUDA", created[1]["backend"])
        self.assertEqual("gpu:3", created[2]["device"])
        self.assertEqual(7, created[2]["cpu_threads"])


if __name__ == "__main__":
    unittest.main()
