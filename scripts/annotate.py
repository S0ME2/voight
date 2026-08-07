#!/usr/bin/env python3
"""Create Uzbekistan passport and ID-card annotation profiles locally."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.annotation import run


def main() -> int:
    parser = argparse.ArgumentParser(description="Guided local document annotation; no models are loaded.")
    parser.add_argument("input", type=Path, help="directory containing passports/ and/or id_cards/")
    parser.add_argument("output", type=Path, help="annotation state and generated profiles directory")
    parser.add_argument("--check", action="store_true", help="validate saved annotations without opening OpenCV windows")
    args = parser.parse_args()
    try:
        return run(args.input, args.output, args.check)
    except ValueError as error:
        parser.error(str(error))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
