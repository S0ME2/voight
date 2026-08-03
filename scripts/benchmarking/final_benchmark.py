from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

from scripts.benchmarking.benchmark import Config, load_samples, rank_key, run_stage, write_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the final comparison of the three selected PaddleOCR MRZ configurations."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("./dataset"),
        help="Dataset root containing labels and generated manifest.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("./outputs/benchmarks/ocr"),
        help="Root directory where the timestamped final benchmark is saved.",
    )
    parser.add_argument("--device", default="cpu", help="PaddleOCR device, for example cpu or gpu:0.")
    parser.add_argument(
        "--max-side",
        type=int,
        default=3000,
        help="Maximum image side used during benchmark preprocessing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_samples(
        dataset_root=args.dataset_root,
        labels_path=None,
        manifest_path=None,
    )

    output_dir = args.output_root / f"final_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)

    configs = [
        Config(
            stage="final_comparison",
            model_name="v6_medium_gray_ori_off_box060",
            detection_model="PP-OCRv6_medium_det",
            recognition_model="PP-OCRv6_medium_rec",
            preprocessing="gray",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_thresh=0.30,
            text_det_box_thresh=0.60,
            text_det_unclip_ratio=2.00,
        ),
        Config(
            stage="final_comparison",
            model_name="v6_medium_gray_ori_on_box060",
            detection_model="PP-OCRv6_medium_det",
            recognition_model="PP-OCRv6_medium_rec",
            preprocessing="gray",
            use_doc_orientation_classify=True,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_thresh=0.30,
            text_det_box_thresh=0.60,
            text_det_unclip_ratio=2.00,
        ),
        Config(
            stage="final_comparison",
            model_name="v6_medium_gray_ori_off_box050",
            detection_model="PP-OCRv6_medium_det",
            recognition_model="PP-OCRv6_medium_rec",
            preprocessing="gray",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_det_thresh=0.30,
            text_det_box_thresh=0.50,
            text_det_unclip_ratio=2.00,
        ),
    ]

    summaries, predictions = run_stage(
        name="final_comparison",
        configs=configs,
        samples=samples,
        device=args.device,
        max_side=args.max_side,
        output_dir=output_dir,
    )

    ranked = sorted(summaries, key=rank_key, reverse=True)
    write_csv(output_dir / "final_ranking.csv", ranked)
    write_csv(
        output_dir / "final_predictions.csv",
        [asdict(prediction) for prediction in predictions],
    )

    print("\n================ FINAL RESULT ================")
    for index, result in enumerate(ranked, start=1):
        print(
            f"{index}. {result['model_name']} | "
            f"exact={result['full_mrz_exact_match_rate']:.1%} | "
            f"char={result['mean_char_accuracy']:.1%} | "
            f"mean={result['mean_latency_seconds']:.3f}s | "
            f"checkdigit coverage={result['checkdigit_gate_coverage']:.1%} | "
            f"checkdigit precision={result['checkdigit_gate_exact_precision']}"
        )
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
