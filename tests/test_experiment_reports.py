"""Small structural checks for the versioned experiment reports."""

import re
import unittest
from pathlib import Path


ROOT = Path(__file__).parents[1]
REPORTS = sorted((ROOT / "experiments").glob("0[1-8].*.md"))


class ExperimentReportTests(unittest.TestCase):
    def test_reports_are_independent(self):
        names = {path.name for path in REPORTS}
        self.assertEqual(len(REPORTS), 8)
        for report in REPORTS:
            text = report.read_text(encoding="utf-8")
            self.assertTrue(
                {"## Metadata", "## 3. Results", "## 9. Reproduction"}
                <= set(re.findall(r"^## .+$", text, re.MULTILINE))
            )
            cross_refs = [
                name
                for name in names
                if name != report.name
                and re.search(rf"\]\((?:\./)?{re.escape(name)}(?:#[^)]+)?\)", text)
            ]
            self.assertFalse(cross_refs, f"{report.name} links to other reports: {cross_refs}")

    def test_report_figures_exist(self):
        for report in REPORTS:
            text = report.read_text(encoding="utf-8")
            for asset in re.findall(r"!\[[^]]*\]\((assets/[^)]+)\)", text):
                self.assertTrue((report.parent / asset).is_file(), f"missing {asset} for {report.name}")
