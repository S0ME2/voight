"""Launch app.main with the checked-in pre-conservative matcher."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from benchmarks.maintained import pre_conservative_verification as module


def main() -> None:
    sys.modules["app.verification"] = module

    class CompatibleVerificationLine:
        def __init__(self, text, confidence, source=None, *_ignored):
            self.text = text
            self.confidence = confidence
            self.source = source

    module.VerificationLine = CompatibleVerificationLine
    old_verify_fields = module.verify_fields

    def verify_fields(lines, expected_fields, thresholds=module.DEFAULT_THRESHOLDS, *, document_type=None, instrumentation=None):
        return old_verify_fields(lines, expected_fields, thresholds, instrumentation=instrumentation)

    module.verify_fields = verify_fields
    import uvicorn

    uvicorn.run("app.main:app", host="127.0.0.1", port=int(sys.argv[1]), workers=1)


if __name__ == "__main__":
    main()
