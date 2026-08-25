from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from app.config import ModelSettings, Settings
from scripts.models.prepare import required_paddle_models, verify_models


class ModelProvisioningTests(unittest.TestCase):
    def test_expected_paths_and_missing_cache_error(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            settings = replace(Settings.from_env(), models=replace(ModelSettings(), directory=root))
            expected = required_paddle_models(settings)
            self.assertEqual(
                [path.relative_to(root).as_posix() for path in expected],
                [
                    "official_models/PP-OCRv6_medium_det",
                    "official_models/latin_PP-OCRv5_mobile_rec",
                ],
            )
            with self.assertRaisesRegex(FileNotFoundError, "missing prepared model directories"):
                verify_models(settings)
            for path in expected:
                path.mkdir(parents=True)
                (path / "model.json").write_text("{}")
            self.assertEqual(expected, verify_models(settings))


if __name__ == "__main__":
    unittest.main()
