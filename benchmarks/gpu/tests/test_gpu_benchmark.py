import json
import os
import tempfile
import unittest
from subprocess import CompletedProcess
from pathlib import Path
from unittest.mock import patch

from benchmarks.gpu.collectors import parse_nvidia_smi
from benchmarks.gpu.gpu_benchmark import arguments, build_configs
from benchmarks.gpu.helpers import compare_semantics, semantic_digest, stats
from benchmarks.gpu.matrix import expand, from_json
from benchmarks.gpu.reporting import summary_rows
from benchmarks.gpu.runtime_guard import ExecutionRefused, require_server_execution


class GpuBenchmarkTests(unittest.TestCase):
    def test_matrix_expands_one_axis_and_rejects_illegal_backend(self):
        rows = expand("recognition-batch", [2, 8])
        self.assertEqual([2, 8], [row.env["TEXT_RECOGNITION_BATCH_SIZE"] for row in rows])
        custom = expand("precision", ["fp16"])[0]
        self.assertIsNone(custom.invalid_reason)

    def test_stats_digest_and_semantic_difference(self):
        self.assertEqual(2, stats([1, 2, 3])["median"])
        before = {"items": [{"success": True, "result": {"fields": {"name": {"value": "A", "confidence": 0.1}}}}]}
        after = {"items": [{"success": True, "result": {"fields": {"name": {"value": "A", "confidence": 0.9}}}}]}
        self.assertEqual(semantic_digest(before), semantic_digest(after))
        differences = compare_semantics(before, after)
        self.assertEqual(0, differences["semantic_change_count"])
        self.assertEqual(1, differences["confidence_only_change_count"])

    def test_fake_nvidia_smi_parser(self):
        rows = parse_nvidia_smi("0, Tesla V100-PCIE-32GB, GPU-uuid, 123, 45, 6, 55, 80\n")
        self.assertEqual("Tesla V100-PCIE-32GB", rows[0]["name"])
        self.assertEqual(123, rows[0]["memory_used_mb"])

    def test_summary_and_explicit_cartesian_are_cpu_only(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "matrix.json"
            path.write_text(json.dumps({"matrix": {"TEXT_RECOGNITION_BATCH_SIZE": [2, 4], "TEXT_RECOGNITION_PACKING": ["fixed-width", "aspect-ratio"]}}), encoding="utf-8")
            self.assertEqual(4, len(from_json(str(path), full_cartesian=True)))
        rows = summary_rows([{"config_id": "a", "axis": "x", "phase": "measured", "latency_seconds": 1, "throughput_per_second": 2, "status": "ok", "cleanup_verified": True}])
        self.assertEqual(1, rows[0]["latency_seconds"])

    def test_guard_refuses_before_gpu_checks_when_ack_missing(self):
        with patch.dict(os.environ, {"RUNTIME_TARGET": "gpu"}, clear=True), patch("benchmarks.gpu.runtime_guard.shutil.which") as which, patch("benchmarks.gpu.runtime_guard.subprocess.run") as run:
            with self.assertRaisesRegex(ExecutionRefused, "VOIGHT_GPU_BENCHMARK_HOST"):
                require_server_execution(execute=True)
            which.assert_not_called()
            run.assert_not_called()

    def test_guard_refuses_cpu_target(self):
        with patch.dict(os.environ, {"RUNTIME_TARGET": "cpu", "VOIGHT_GPU_BENCHMARK_HOST": "1"}, clear=True):
            with self.assertRaisesRegex(ExecutionRefused, "RUNTIME_TARGET"):
                require_server_execution(execute=True)

    def test_guard_refuses_missing_image_after_gpu_visibility_check(self):
        calls = []
        def runner(command, **_):
            calls.append(command)
            if command[0] == "nvidia-smi":
                return CompletedProcess(command, 0, "0, Tesla V100-PCIE-32GB, GPU, 535\n", "")
            return CompletedProcess(command, 1, "", "missing")
        with patch.dict(os.environ, {"RUNTIME_TARGET": "gpu", "VOIGHT_GPU_BENCHMARK_HOST": "1", "GPU_ID": "0"}, clear=True), patch("benchmarks.gpu.runtime_guard.shutil.which", return_value="/usr/bin/tool"):
            with self.assertRaisesRegex(ExecutionRefused, "image"):
                require_server_execution(execute=True, runner=runner)
        self.assertEqual("nvidia-smi", calls[0][0])
        self.assertEqual("docker", calls[1][0])

    def test_plan_parser_does_not_require_server_ack(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "passport").mkdir()
            args = arguments(["--plan", "--experiment", "recognition-batch", "--values", "2,4", "--dataset-root", str(root)])
            self.assertTrue(args.plan)
            self.assertEqual([2, 4], [row.value for row in build_configs(args)])


if __name__ == "__main__":
    unittest.main()
