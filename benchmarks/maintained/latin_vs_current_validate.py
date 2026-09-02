from pathlib import Path
import sys

required = "environment.json latin_config.json current_config.json raw_runs.csv overall_accuracy.csv document_type_accuracy.csv field_comparison.csv document_comparison.csv failure_transitions.csv strict_field_results.csv performance.csv stage_timings.csv resource_usage.csv ocr_differences.csv known_failure_comparison.csv controlled_same_verifier.csv summary.json report.md".split()
root = Path(sys.argv[1])
missing = [name for name in required if not (root / name).is_file() or not (root / name).stat().st_size]
if missing:
    print("missing or empty:", ", ".join(missing))
    raise SystemExit(1)
print("validated", root)
