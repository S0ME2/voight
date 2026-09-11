import hashlib
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from benchmarks.maintained import profile_other_benchmark as benchmark
from benchmarks.maintained.full_dataset_api_batch_benchmark import _records


class _Response:
    ok = True
    text = ""

    def __init__(self, payload):
        self.payload = payload

    def json(self):
        return self.payload


class ProfileOtherBenchmarkTests(unittest.TestCase):
    @contextmanager
    def fixture_records(self, kind):
        folder = {"passport": "passport", "driving-license": "driving_license"}[kind]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / folder / ("p_1.png" if kind == "passport" else "d_1.jpg")
            image.parent.mkdir(parents=True)
            image.write_bytes(b"fixture")
            relative = image.relative_to(root).as_posix()
            annotation = {
                "id": "fixture-1",
                "images": {"image": relative},
                "image_sha256": {"image": hashlib.sha256(image.read_bytes()).hexdigest()},
                "fields": {"surname": {"state": "value", "value": "DOE"}},
            }
            annotation_path = root / "annotations" / folder / "fixture.json"
            annotation_path.parent.mkdir(parents=True)
            annotation_path.write_text(json.dumps(annotation), encoding="utf-8")
            yield root

    def test_full_route_requires_application_profiling(self):
        with self.fixture_records("passport") as root:
            records = _records(root, "passport")
            response = _Response({"diagnostics": {}})
            with patch.object(benchmark.requests, "post", return_value=response):
                with self.assertRaisesRegex(RuntimeError, "VOIGHT_BENCHMARK_PROFILE=true"):
                    benchmark._post_full("http://test", "passport", records, 1, 1)

    def test_comparison_route_runs_ocr_then_check(self):
        calls = []

        def post(url, **kwargs):
            calls.append(url)
            if url.endswith("/ocr/batch"):
                return _Response({"items": [{"success": True, "result": {"lines": []}}]})
            return _Response({"summary": {"match": 1, "likely_match": 0, "mismatch": 0, "not_found": 0}})

        with self.fixture_records("passport") as root:
            records = _records(root, "passport")
            with patch.object(benchmark.requests, "post", side_effect=post):
                row = benchmark._post_comparison("http://test", "passport", records, 1, 1)

        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["comparison_summary"]["match"], 1)
        self.assertEqual([url.rsplit("/", 2)[-2:] for url in calls], [["ocr", "batch"], ["passport", "check"]])

    def test_comparison_route_uses_api_spelling_for_driving_licence(self):
        urls = []

        def post(url, **kwargs):
            urls.append(url)
            if url.endswith("/ocr/batch"):
                return _Response({"items": [{"success": True, "result": {"lines": []}}]})
            return _Response({"summary": {"match": 1, "likely_match": 0, "mismatch": 0, "not_found": 0}})

        with self.fixture_records("driving-license") as root:
            records = _records(root, "driving-license")
            with patch.object(benchmark.requests, "post", side_effect=post):
                benchmark._post_comparison("http://test", "driving-license", records, 1, 1)

        self.assertTrue(all("/verification/driving-licence/" in url for url in urls))


if __name__ == "__main__":
    unittest.main()
