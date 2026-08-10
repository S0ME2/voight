import os
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from app.config import ModelSettings, ProfileSettings, RuntimeSettings, Settings


class SettingsTests(unittest.TestCase):
    def test_defaults_are_valid_and_cpu_only(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = Settings.from_env()
        self.assertEqual(settings.runtime.target, "cpu")
        self.assertEqual(settings.ocr.device, "cpu")
        self.assertGreater(settings.runtime.cpu_threads, 0)

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


if __name__ == "__main__":
    unittest.main()
