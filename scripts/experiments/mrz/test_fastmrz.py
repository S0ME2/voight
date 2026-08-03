from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import cv2
from fastmrz import FastMRZ


# =========================================================
# DEFAULT CONFIG
# =========================================================

DEFAULT_IMAGE = Path("assets/samples/passport/uzpassport.png")

DEFAULT_OUTPUT_ROOT = Path("outputs/experiments/mrz/fastmrz")


# =========================================================
# JSON HELPERS
# =========================================================


def save_json(
    path: Path,
    data: Any,
) -> None:
    """
    Save JSON with UTF-8 encoding.
    """

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            ensure_ascii=False,
            indent=4,
            default=str,
        )


# =========================================================
# TESSERACT VALIDATION
# =========================================================


def get_tesseract_languages(
    tesseract_path: str,
) -> set[str]:
    """
    Return all installed Tesseract languages.
    """

    result = subprocess.run(
        [
            tesseract_path,
            "--list-langs",
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    # Depending on Tesseract version, output may be
    # written to stdout or stderr.
    output = result.stdout + "\n" + result.stderr

    languages = set()

    for line in output.splitlines():
        line = line.strip()

        if not line:
            continue

        # Ignore informational lines such as:
        #
        # List of available languages in ...
        if line.lower().startswith("list of available languages"):
            continue

        languages.add(line)

    return languages


def verify_tesseract(
    explicit_path: str | None,
) -> str:
    """
    Verify that:

        1. Tesseract exists.
        2. The custom 'mrz' language exists.

    Returns the resolved Tesseract executable path.
    """

    if explicit_path:
        tesseract_path = explicit_path
    else:
        discovered = shutil.which("tesseract")

        if discovered is None:
            raise RuntimeError(
                "Tesseract was not found in PATH.\n"
                "\n"
                "Install it on Ubuntu/Debian with:\n"
                "\n"
                "sudo apt install tesseract-ocr"
            )

        tesseract_path = discovered

    languages = get_tesseract_languages(tesseract_path)

    if "mrz" not in languages:
        raise RuntimeError(
            "Tesseract is installed, but the "
            "'mrz' language data is missing.\n"
            "\n"
            "FastMRZ requires mrz.traineddata "
            "to be installed in the Tesseract "
            "tessdata directory."
        )

    return tesseract_path


# =========================================================
# FASTMRZ INITIALIZATION
# =========================================================


def create_fastmrz(
    tesseract_path: str | None,
    tessdata_path: str | None,
) -> FastMRZ:
    """
    Initialize FastMRZ.

    Only pass optional arguments when the user
    explicitly provided them.

    This avoids passing empty strings to FastMRZ.
    """

    kwargs = {}

    if tesseract_path:
        kwargs["tesseract_path"] = tesseract_path

    if tessdata_path:
        kwargs["tessdata_path"] = tessdata_path

    return FastMRZ(**kwargs)


# =========================================================
# IMAGE LOADING
# =========================================================


def load_color_image(
    image_path: Path,
):
    """
    Load image explicitly as a 3-channel BGR image.

    This is important because FastMRZ's preprocessing
    expects three channels before reshaping its model
    input to:

        (1, 256, 256, 3)

    Passing the NumPy image directly also avoids the
    file-path conversion issue that previously produced
    a single-channel 256x256 array.
    """

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(f"OpenCV could not read: {image_path}")

    if image.ndim != 3:
        raise RuntimeError(f"Expected a 3-dimensional image array, got: {image.shape}")

    if image.shape[2] != 3:
        raise RuntimeError(f"Expected a 3-channel image, got shape: {image.shape}")

    return image


# =========================================================
# RAW MRZ NORMALIZATION
# =========================================================


def normalize_raw_mrz(
    raw_result: Any,
) -> str:
    """
    Convert FastMRZ's raw output into a clean string
    for saving and displaying.

    Normally ignore_parse=True returns a string.
    This function is defensive in case a future version
    returns another representation.
    """

    if raw_result is None:
        return ""

    if isinstance(
        raw_result,
        str,
    ):
        return raw_result.strip()

    if isinstance(
        raw_result,
        list,
    ):
        return "\n".join(str(item).strip() for item in raw_result if str(item).strip())

    if isinstance(
        raw_result,
        dict,
    ):
        mrz_text = raw_result.get("mrz_text")

        if mrz_text:
            return str(mrz_text).strip()

    return str(raw_result).strip()


# =========================================================
# MRZ DIAGNOSTICS
# =========================================================


def analyze_raw_mrz(
    raw_mrz_text: str,
) -> dict:
    """
    Generate simple structural diagnostics.

    For a normal TD3 passport, we expect:

        2 lines
        44 characters per line

    This does NOT modify or repair the OCR output.
    """

    lines = [line.strip() for line in raw_mrz_text.splitlines() if line.strip()]

    return {
        "line_count": len(lines),
        "lines": [
            {
                "text": line,
                "length": len(line),
            }
            for line in lines
        ],
        "looks_like_td3": (len(lines) == 2 and all(len(line) == 44 for line in lines)),
    }


# =========================================================
# ARGUMENTS
# =========================================================


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Test FastMRZ end-to-end extraction and inspect raw MRZ OCR output."
        )
    )

    parser.add_argument(
        "image",
        nargs="?",
        default=str(DEFAULT_IMAGE),
        help=("Passport/document image path."),
    )

    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help=("Root directory for experiment outputs."),
    )

    parser.add_argument(
        "--tesseract-path",
        default=None,
        help=("Optional explicit path to the Tesseract executable."),
    )

    parser.add_argument(
        "--tessdata-path",
        default=None,
        help=("Optional explicit path to the Tesseract tessdata directory."),
    )

    return parser.parse_args()


# =========================================================
# MAIN
# =========================================================


def main():
    args = parse_args()

    image_path = Path(args.image)

    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    output_dir = Path(args.output_root) / image_path.stem

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("=" * 70)
    print("FASTMRZ TEST")
    print("=" * 70)

    print(f"Input:  {image_path}")

    print(f"Output: {output_dir}")

    # =====================================================
    # 1. VERIFY TESSERACT
    # =====================================================

    resolved_tesseract_path = verify_tesseract(args.tesseract_path)

    print()
    print(f"Tesseract: {resolved_tesseract_path}")

    # =====================================================
    # 2. LOAD IMAGE AS 3-CHANNEL NUMPY ARRAY
    # =====================================================

    image = load_color_image(image_path)

    image_height, image_width = image.shape[:2]

    print(f"Image size: {image_width}x{image_height}")

    print(f"Image shape: {image.shape}")

    input_copy_path = output_dir / "01_input.jpg"

    if not cv2.imwrite(
        str(input_copy_path),
        image,
    ):
        raise RuntimeError(f"Could not save: {input_copy_path}")

    # =====================================================
    # 3. INITIALIZE FASTMRZ
    # =====================================================

    init_start = time.perf_counter()

    fast_mrz = create_fastmrz(
        tesseract_path=(args.tesseract_path),
        tessdata_path=(args.tessdata_path),
    )

    init_seconds = time.perf_counter() - init_start

    # =====================================================
    # 4. GET RAW MRZ TEXT
    # =====================================================
    #
    # IMPORTANT:
    #
    # ignore_parse=True allows us to inspect the
    # recognition result BEFORE FastMRZ tries to
    # parse it as TD1 / TD2 / TD3.
    #
    # We pass:
    #
    #     image
    #     input_type="numpy"
    #
    # instead of the file path to avoid the previous
    # single-channel input problem.
    # =====================================================

    raw_start = time.perf_counter()

    try:
        raw_result = fast_mrz.get_details(
            image,
            input_type="numpy",
            ignore_parse=True,
        )

        raw_error = None

    except Exception as error:
        raw_result = None

        raw_error = f"{type(error).__name__}: {error}"

    raw_extraction_seconds = time.perf_counter() - raw_start

    raw_mrz_text = normalize_raw_mrz(raw_result)

    # =====================================================
    # 5. ANALYZE RAW MRZ
    # =====================================================

    raw_analysis = analyze_raw_mrz(raw_mrz_text)

    raw_text_path = output_dir / "raw_mrz.txt"

    with raw_text_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(raw_mrz_text)

    save_json(
        output_dir / "raw_mrz.json",
        {
            "raw_mrz_text": (raw_mrz_text),
            "error": (raw_error),
            "analysis": (raw_analysis),
        },
    )

    # =====================================================
    # 6. RUN NORMAL FASTMRZ PARSING
    # =====================================================
    #
    # This intentionally performs a separate normal
    # FastMRZ run.
    #
    # Why?
    #
    # It lets us compare:
    #
    #     RAW OCR RESULT
    #
    # against:
    #
    #     FINAL PARSED RESULT
    #
    # without relying on FastMRZ private/internal APIs.
    #
    # parsed_pipeline_seconds is therefore the timing
    # you should compare with other end-to-end solutions.
    # =====================================================

    parsed_start = time.perf_counter()

    try:
        parsed_details = fast_mrz.get_details(
            image,
            input_type="numpy",
            include_checkdigit=False,
        )

        parsed_error = None

    except Exception as error:
        parsed_details = {
            "status": "EXCEPTION",
            "status_message": (str(error)),
        }

        parsed_error = f"{type(error).__name__}: {error}"

    parsed_pipeline_seconds = time.perf_counter() - parsed_start

    # =====================================================
    # 7. SAVE COMBINED RESULT
    # =====================================================

    combined_result = {
        "raw_mrz_text": (raw_mrz_text),
        "raw_mrz_analysis": (raw_analysis),
        "parsed": (parsed_details),
        "errors": {
            "raw_extraction": (raw_error),
            "parsed_pipeline": (parsed_error),
        },
    }

    save_json(
        output_dir / "result.json",
        combined_result,
    )

    # =====================================================
    # 8. SAVE TIMINGS
    # =====================================================

    timings = {
        # One-time model/object initialization.
        "init_seconds": (init_seconds),
        # Diagnostic raw extraction:
        # detection + OCR, parser skipped.
        "raw_extraction_seconds": (raw_extraction_seconds),
        # Normal FastMRZ end-to-end run:
        # localization/detection + OCR + parsing.
        #
        # USE THIS NUMBER when comparing FastMRZ
        # against another complete extraction pipeline.
        "parsed_pipeline_seconds": (parsed_pipeline_seconds),
    }

    save_json(
        output_dir / "timings.json",
        timings,
    )

    # =====================================================
    # 9. PRINT RAW MRZ
    # =====================================================

    print()
    print("=" * 70)
    print("FASTMRZ RAW MRZ")
    print("=" * 70)

    if raw_error:
        print(f"ERROR: {raw_error}")

    elif raw_mrz_text:
        print(raw_mrz_text)

    else:
        print("<EMPTY>")

    # =====================================================
    # 10. PRINT RAW MRZ ANALYSIS
    # =====================================================

    print()
    print("=" * 70)
    print("RAW MRZ ANALYSIS")
    print("=" * 70)

    print(
        json.dumps(
            raw_analysis,
            ensure_ascii=False,
            indent=4,
        )
    )

    # =====================================================
    # 11. PRINT PARSED RESULT
    # =====================================================

    print()
    print("=" * 70)
    print("FASTMRZ PARSED")
    print("=" * 70)

    print(
        json.dumps(
            parsed_details,
            ensure_ascii=False,
            indent=4,
            default=str,
        )
    )

    # =====================================================
    # 12. PRINT TIMINGS
    # =====================================================

    print()
    print("=" * 70)
    print("TIMINGS")
    print("=" * 70)

    print(
        json.dumps(
            timings,
            indent=4,
        )
    )

    # =====================================================
    # 13. PRINT OUTPUTS
    # =====================================================

    print()
    print("=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(f"Input copy:    {input_copy_path}")

    print(f"Raw MRZ text:  {raw_text_path}")

    print(f"Raw MRZ JSON:  {output_dir / 'raw_mrz.json'}")

    print(f"Full result:   {output_dir / 'result.json'}")

    print(f"Timings:       {output_dir / 'timings.json'}")


if __name__ == "__main__":
    main()
