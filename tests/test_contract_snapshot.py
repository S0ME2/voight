import json
import os
import unittest
from unittest import mock
from pathlib import Path

from app.config import Settings
from app.main import create_app


ROOT = Path(__file__).resolve().parents[1]


class ContractSnapshotTests(unittest.TestCase):
    def test_openapi_is_the_committed_contract(self):
        with mock.patch.dict(os.environ, {"MODEL_DIR": ""}, clear=False):
            actual = create_app(Settings.from_env()).openapi()
        expected = json.loads((ROOT / "tests/golden/openapi.json").read_text())
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
