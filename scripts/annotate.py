#!/usr/bin/env python3
"""Create all supported document annotations from JSON-defined layouts."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.annotation import DEFAULT_LAYOUTS, run


def main() -> int:
    parser = argparse.ArgumentParser(description="Guided local document annotation; passport mode loads the CPU MRZ detector.")
    parser.add_argument("input", type=Path, help="directory structured according to config/annotation_layouts.json")
    parser.add_argument("output", type=Path, help="annotation state and generated profiles directory")
    parser.add_argument("--check", action="store_true", help="validate saved annotations without opening OpenCV windows")
    parser.add_argument("--layouts", type=Path, default=DEFAULT_LAYOUTS, help="JSON layout definition (default: config/annotation_layouts.json)")
    args = parser.parse_args()
    try:
        return run(args.input, args.output, args.check, args.layouts)
    except ValueError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
