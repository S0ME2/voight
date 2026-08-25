import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from app.config import ModelSettings, ProfileSettings, RuntimeSettings, Settings
from app.models import Models


class SettingsTests(unittest.TestCase):
    def test_defaults_are_valid_and_cpu_only(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.runtime.target, "cpu")
        self.assertEqual(settings.ocr.device, "cpu")
        self.assertGreater(settings.runtime.cpu_threads, 0)
        self.assertEqual(4, settings.runtime.localization_batch_size)
        self.assertEqual(1, settings.runtime.text_detection_batch_size)
        self.assertEqual(2, settings.runtime.text_recognition_batch_size)
        self.assertEqual(2, settings.runtime.mrz_recognition_batch_size)
        self.assertEqual("fixed-width", settings.runtime.text_recognition_packing)
        self.assertEqual(1.50, settings.mrz.contrast)

    def test_runtime_and_device_must_match(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()
        invalid = replace(settings, runtime=replace(settings.runtime, target="gpu"))
        with self.assertRaisesRegex(ValueError, "must select the same runtime"):
            invalid.validate_startup()

    def test_runtime_target_is_primary_and_gpu_id_is_validated(self):
        with patch.dict(os.environ, {"RUNTIME_TARGET": "gpu", "GPU_ID": "2"}, clear=True):
            settings = Settings.from_env()
        self.assertEqual("gpu", settings.ocr.device)
        self.assertEqual(2, settings.runtime.gpu_id)
        with self.assertRaisesRegex(ValueError, "GPU_ID"):
            replace(settings, runtime=replace(settings.runtime, gpu_id=-1)).validate_startup()

    def test_missing_profiles_and_explicit_model_directory_fail_clearly(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()
        missing = Path("/definitely/missing/voight-profile.json")
        invalid_profile = replace(
            settings,
            profiles=ProfileSettings(missing, settings.profiles.id_card),
        )
        with self.assertRaisesRegex(ValueError, "PASSPORT_PROFILE does not exist"):
            invalid_profile.validate_startup()
        invalid_model = replace(settings, models=ModelSettings(missing))
        with self.assertRaisesRegex(ValueError, "MODEL_DIR does not exist"):
            invalid_model.validate_startup()

    def test_env_controls_batches_queue_threads_paths_and_uploads(self):
        with tempfile.TemporaryDirectory() as model_dir, patch.dict(
            os.environ,
            {
                "CPU_THREADS": "2",
                "REQUEST_QUEUE_LIMIT": "5",
                "LOCALIZATION_BATCH_SIZE": "2",
                "TEXT_DETECTION_BATCH_SIZE": "3",
                "TEXT_RECOGNITION_BATCH_SIZE": "4",
                "TEXT_RECOGNITION_PROCESSES": "2",
                "BATCH_MAX_FILES": "6",
                "BATCH_MAX_FILE_BYTES": "100",
                "BATCH_MAX_ARCHIVE_UNCOMPRESSED_BYTES": "200",
                "MODEL_DIR": model_dir,
                "PRELOAD": "true",
                "LOGGING": "false",
            },
            clear=True,
        ):
            settings = Settings.from_env()
        self.assertEqual(settings.runtime, RuntimeSettings("cpu", 2, 5, 2, 3, 4, 2))
        self.assertEqual(settings.batch.max_files, 6)
        self.assertEqual(settings.batch.max_file_bytes, 100)
        self.assertTrue(settings.preload)
        self.assertFalse(settings.artifacts.enabled)

    def test_multiple_recognition_processes_are_cpu_only(self):
        with patch.dict(
            os.environ,
            {"RUNTIME_TARGET": "gpu", "TEXT_RECOGNITION_PROCESSES": "2"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "only on CPU"):
                Settings.from_env()

    def test_gpu_recognition_acceleration_requires_gpu_and_valid_precision(self):
        with patch.dict(os.environ, {"TEXT_RECOGNITION_PRECISION": "fp16"}, clear=True):
            with self.assertRaisesRegex(ValueError, "require RUNTIME_TARGET=gpu"):
                Settings.from_env()
        with patch.dict(os.environ, {"RUNTIME_TARGET": "gpu", "TEXT_RECOGNITION_PRECISION": "int8"}, clear=True):
            with self.assertRaisesRegex(ValueError, "must be 'fp32' or 'fp16'"):
                Settings.from_env()
        with patch.dict(
            os.environ,
            {
                "RUNTIME_TARGET": "gpu",
                "TEXT_RECOGNITION_ENABLE_HPI": "true",
                "TEXT_RECOGNITION_USE_TENSORRT": "true",
                "TEXT_RECOGNITION_PRECISION": "fp16",
            },
            clear=True,
        ):
            settings = Settings.from_env()
        self.assertTrue(settings.runtime.text_recognition_enable_hpi)
        self.assertTrue(settings.runtime.text_recognition_use_tensorrt)
        self.assertEqual("fp16", settings.runtime.text_recognition_precision)

    def test_model_backends_and_names_are_selected_in_one_place(self):
        with patch.dict(
            os.environ,
            {
                "TEXT_DETECTOR_BACKEND": "paddle",
                "TEXT_DETECTOR_MODEL": "detector-x",
                "TEXT_RECOGNIZER_BACKEND": "custom",
                "TEXT_RECOGNIZER_MODEL": "recognizer-y",
                "DOCUMENT_LOCALIZER_BACKEND": "document-z",
                "MRZ_LOCALIZER_BACKEND": "mrz-localizer-z",
                "MRZ_RECOGNIZER_BACKEND": "mrz-recognizer-z",
            },
            clear=True,
        ):
            settings = Settings.from_env()
        self.assertEqual("detector-x", settings.models.text_detector.model)
        self.assertEqual("custom", settings.models.text_recognizer.backend)
        self.assertEqual("recognizer-y", settings.models.text_recognizer.model)
        self.assertEqual("document-z", settings.models.localization.document_backend)
        self.assertEqual("mrz-localizer-z", settings.models.localization.mrz_backend)
        self.assertEqual("mrz-recognizer-z", settings.models.mrz.recognizer_backend)

    def test_detector_resize_settings_are_typed_and_reported(self):
        with patch.dict(
            os.environ,
            {"TEXT_DETECTOR_PIXEL_SCALE": "0.5", "TEXT_DETECTOR_LIMIT_SIDE_LEN": "640"},
            clear=True,
        ):
            settings = Settings.from_env()
        self.assertEqual(0.5, settings.runtime.text_detector_pixel_scale)
        self.assertEqual(640, settings.runtime.text_detector_limit_side_len)
        resize = Models(settings).configuration()["text_detector"]["resize"]
        self.assertEqual(25.0, resize["effective_percent"])
        self.assertEqual(640, resize["limit_side_len_override"])

    def test_detector_resize_settings_validate_at_startup(self):
        with patch.dict(os.environ, {"TEXT_DETECTOR_PIXEL_SCALE": "0"}, clear=True):
            with self.assertRaisesRegex(ValueError, "TEXT_DETECTOR_PIXEL_SCALE"):
                Settings.from_env()


if __name__ == "__main__":
    unittest.main()
