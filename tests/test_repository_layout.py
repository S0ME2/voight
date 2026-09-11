import subprocess
import unittest
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PATH = re.compile(r"/Users/|/home/(?!voight(?:/|\b))|(?:^|[\"'])~/(?:Desktop/)?|[A-Za-z]:\\\\")


def tracked_files() -> list[str]:
    return subprocess.run(
        ["git", "ls-files"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()


class RepositoryLayoutTests(unittest.TestCase):
    def test_onboarding_files_exist_and_legacy_paths_are_gone(self):
        required = {
            "README.md",
            "setup.sh",
            ".env.example",
            "docs/getting-started.md",
            "docs/architecture.md",
            "docs/api.md",
            "docs/configuration.md",
            "docs/models.md",
            "docs/deployment.md",
            "docs/development.md",
            "docs/benchmarking.md",
            "docs/dataset.md",
            "docs/reorganization-2026-08.md",
        }
        self.assertFalse([path for path in required if not (ROOT / path).is_file()])
        removed = (
            "app/api/routes.py",
            "app/api/batch.py",
            "app/workflows.py",
            "scripts/pipelines/driving_license_pipeline.py",
            "complexity/benchmark_batch_complexity.py",
            "GATES.md",
            "scripts/benchmarking/pipeline_breakdown.py",
            "scripts/experiments/mrz/annotate_passport_mrz.py",
        )
        self.assertFalse([path for path in removed if (ROOT / path).exists()])

    def test_benchmarks_are_separated_into_maintained_and_historical(self):
        maintained = ROOT / "benchmarks" / "maintained"
        historical = ROOT / "benchmarks" / "historical"
        self.assertTrue(maintained.is_dir())
        self.assertTrue(historical.is_dir())
        # Core maintained benchmarks stay runnable through Makefile targets.
        for name in (
            "benchmark_batch_complexity.py",
            "pipeline_breakdown.py",
            "text_recognition_batch_benchmark.py",
        ):
            self.assertTrue((maintained / name).is_file(), name)
        # Historical experiment drivers are preserved under their own area.
        for name in (
            "cpu_thread_benchmark.py",
            "recognizer_ab_benchmark.py",
            "mrz_preprocessing_reconciliation.py",
        ):
            self.assertTrue((historical / name).is_file(), name)

    def test_tracked_files_avoid_legacy_tool_locations(self):
        legacy_prefixes = ("scripts/benchmarking/", "scripts/experiments/", "complexity/")
        offenders = [path for path in tracked_files() if path.startswith(legacy_prefixes)]
        self.assertEqual([], offenders)

    def test_exploration_prototypes_are_archived_not_deleted(self):
        archived = ROOT / "archive" / "scripts-experiments"
        self.assertTrue(archived.is_dir())
        for name in (
            "alignment/test_docaligner.py",
            "mrz/test_fastmrz.py",
            "ocr/best_run.py",
            "roi/test_rois.py",
        ):
            self.assertTrue((archived / name).is_file(), name)

    def test_cpu_and_gpu_dependency_sets_stay_separate(self):
        cpu_lock = (ROOT / "requirements" / "cpu.lock").read_text(encoding="utf-8")
        gpu_requirements = (ROOT / "requirements" / "gpu.txt").read_text(encoding="utf-8")

        for package in ("docaligner-docsaid", "mrzscanner-docsaid", "onnxruntime"):
            self.assertRegex(cpu_lock, rf"(?m)^{re.escape(package)}==")
        self.assertNotRegex(cpu_lock, r"(?m)^(onnxruntime-gpu|paddlepaddle-gpu)==")
        gpu_lock = (ROOT / "requirements" / "gpu.lock").read_text(encoding="utf-8")
        self.assertRegex(gpu_lock, r"(?m)^onnxruntime-gpu==1\.22\.0")
        self.assertIn("paddlepaddle_gpu-3.2.2", gpu_lock)
        self.assertIn("-r gpu.lock", gpu_requirements)

    def test_tracked_project_text_has_no_developer_absolute_paths(self):
        offenders = []
        for relative in subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines():
            path = ROOT / relative
            if relative.startswith(".codex/") or relative == "tests/test_repository_layout.py" or not path.is_file():
                continue
            if path.suffix not in {".py", ".md", ".toml", ".yaml", ".yml", ".txt", ".json"} and path.name not in {"Dockerfile", "Makefile", ".env.example"}:
                continue
            if LOCAL_PATH.search(path.read_text(encoding="utf-8", errors="ignore")):
                offenders.append(relative)
        self.assertEqual([], offenders)


if __name__ == "__main__":
    unittest.main()
