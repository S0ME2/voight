from pathlib import Path
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
LOCAL_PATH = re.compile(r"/Users/|/home/(?!voight(?:/|\b))|(?:^|[\"'])~/(?:Desktop/)?|[A-Za-z]:\\\\")


class RepositoryLayoutTests(unittest.TestCase):
    def test_onboarding_files_exist_and_legacy_paths_are_gone(self):
        required = {
            "README.md",
            ".env.example",
            "docs/architecture.md",
            "docs/api.md",
            "docs/configuration.md",
            "docs/models.md",
            "docs/deployment.md",
            "docs/development.md",
            "docs/benchmarking.md",
            "docs/dataset.md",
        }
        self.assertFalse([path for path in required if not (ROOT / path).is_file()])
        removed = (
            "app/api/routes.py",
            "app/api/batch.py",
            "app/workflows.py",
            "scripts/pipelines/driving_license_pipeline.py",
        )
        self.assertFalse([path for path in removed if (ROOT / path).exists()])

    def test_tracked_project_text_has_no_developer_absolute_paths(self):
        tracked = subprocess.run(
            ["git", "ls-files", "-co", "--exclude-standard"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
        offenders = []
        for relative in tracked:
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
