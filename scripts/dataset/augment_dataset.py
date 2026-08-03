from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import cv2
import numpy as np


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def rotate(
    image: np.ndarray,
    angle: float,
) -> np.ndarray:
    height, width = image.shape[:2]

    matrix = cv2.getRotationMatrix2D(
        (width / 2, height / 2),
        angle,
        1.0,
    )

    return cv2.warpAffine(
        image,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
    )


def perspective_distortion(
    image: np.ndarray,
    strength: float,
    rng: random.Random,
) -> np.ndarray:
    """
    Apply a mild random perspective distortion.

    strength:
        0.01 = very mild
        0.05 = noticeable
        0.10 = strong
    """
    height, width = image.shape[:2]

    max_dx = width * strength
    max_dy = height * strength

    source = np.float32(
        [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1],
        ]
    )

    destination = np.float32(
        [
            [
                rng.uniform(0, max_dx),
                rng.uniform(0, max_dy),
            ],
            [
                width - 1 - rng.uniform(0, max_dx),
                rng.uniform(0, max_dy),
            ],
            [
                width - 1 - rng.uniform(0, max_dx),
                height - 1 - rng.uniform(0, max_dy),
            ],
            [
                rng.uniform(0, max_dx),
                height - 1 - rng.uniform(0, max_dy),
            ],
        ]
    )

    matrix = cv2.getPerspectiveTransform(source, destination)

    return cv2.warpPerspective(
        image,
        matrix,
        (width, height),
        borderMode=cv2.BORDER_REPLICATE,
    )


def adjust_brightness_contrast(
    image: np.ndarray,
    brightness: int,
    contrast: float,
) -> np.ndarray:
    """
    brightness: roughly -100..100
    contrast: 0.5..1.5
    """
    return cv2.convertScaleAbs(
        image,
        alpha=contrast,
        beta=brightness,
    )


def gaussian_blur(
    image: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1

    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        0,
    )


def motion_blur(
    image: np.ndarray,
    kernel_size: int,
    horizontal: bool,
) -> np.ndarray:
    kernel = np.zeros(
        (kernel_size, kernel_size),
        dtype=np.float32,
    )

    if horizontal:
        kernel[kernel_size // 2, :] = 1
    else:
        kernel[:, kernel_size // 2] = 1

    kernel /= kernel_size

    return cv2.filter2D(
        image,
        -1,
        kernel,
    )


def add_noise(
    image: np.ndarray,
    sigma: float,
    np_rng: np.random.Generator,
) -> np.ndarray:
    noise = np_rng.normal(
        0,
        sigma,
        image.shape,
    )

    noisy = image.astype(np.float32) + noise

    return np.clip(
        noisy,
        0,
        255,
    ).astype(np.uint8)


def downscale_and_restore(
    image: np.ndarray,
    scale: float,
) -> np.ndarray:
    """
    Simulates a passport occupying fewer useful pixels.
    """
    height, width = image.shape[:2]

    small_width = max(1, int(width * scale))
    small_height = max(1, int(height * scale))

    small = cv2.resize(
        image,
        (small_width, small_height),
        interpolation=cv2.INTER_AREA,
    )

    return cv2.resize(
        small,
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )


def jpeg_compression(
    image: np.ndarray,
    quality: int,
) -> np.ndarray:
    success, encoded = cv2.imencode(
        ".jpg",
        image,
        [cv2.IMWRITE_JPEG_QUALITY, quality],
    )

    if not success:
        raise RuntimeError("JPEG encoding failed")

    decoded = cv2.imdecode(
        encoded,
        cv2.IMREAD_COLOR,
    )

    if decoded is None:
        raise RuntimeError("JPEG decoding failed")

    return decoded


def grayscale_to_bgr(
    image: np.ndarray,
) -> np.ndarray:
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    # Convert back to 3 channels so all generated files
    # have a consistent image representation.
    return cv2.cvtColor(
        gray,
        cv2.COLOR_GRAY2BGR,
    )


def random_augmentation(
    image: np.ndarray,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> tuple[np.ndarray, dict[str, Any]]:
    """
    Create ONE random augmented image.

    We deliberately apply only 1-3 transformations.
    Applying every possible degradation simultaneously
    often creates unrealistic images.
    """
    result = image.copy()
    operations: list[dict[str, Any]] = []

    available_operations = [
        "rotation",
        "perspective",
        "brightness_contrast",
        "gaussian_blur",
        "motion_blur",
        "noise",
        "downscale",
        "jpeg",
        "grayscale",
    ]

    operation_count = rng.randint(1, 3)

    selected = rng.sample(
        available_operations,
        k=operation_count,
    )

    for operation in selected:
        if operation == "rotation":
            angle = rng.uniform(-15, 15)

            result = rotate(
                result,
                angle,
            )

            operations.append(
                {
                    "type": "rotation",
                    "angle": round(angle, 2),
                }
            )

        elif operation == "perspective":
            strength = rng.uniform(0.01, 0.08)

            result = perspective_distortion(
                result,
                strength,
                rng,
            )

            operations.append(
                {
                    "type": "perspective",
                    "strength": round(strength, 3),
                }
            )

        elif operation == "brightness_contrast":
            brightness = rng.randint(-60, 60)
            contrast = rng.uniform(0.65, 1.35)

            result = adjust_brightness_contrast(
                result,
                brightness,
                contrast,
            )

            operations.append(
                {
                    "type": "brightness_contrast",
                    "brightness": brightness,
                    "contrast": round(contrast, 2),
                }
            )

        elif operation == "gaussian_blur":
            kernel_size = rng.choice([3, 5, 7])

            result = gaussian_blur(
                result,
                kernel_size,
            )

            operations.append(
                {
                    "type": "gaussian_blur",
                    "kernel_size": kernel_size,
                }
            )

        elif operation == "motion_blur":
            kernel_size = rng.choice([3, 5, 7, 9])
            horizontal = rng.choice([True, False])

            result = motion_blur(
                result,
                kernel_size,
                horizontal,
            )

            operations.append(
                {
                    "type": "motion_blur",
                    "kernel_size": kernel_size,
                    "direction": ("horizontal" if horizontal else "vertical"),
                }
            )

        elif operation == "noise":
            sigma = rng.uniform(3, 15)

            result = add_noise(
                result,
                sigma,
                np_rng,
            )

            operations.append(
                {
                    "type": "noise",
                    "sigma": round(sigma, 2),
                }
            )

        elif operation == "downscale":
            scale = rng.uniform(0.35, 0.8)

            result = downscale_and_restore(
                result,
                scale,
            )

            operations.append(
                {
                    "type": "downscale",
                    "scale": round(scale, 2),
                }
            )

        elif operation == "jpeg":
            quality = rng.randint(35, 85)

            result = jpeg_compression(
                result,
                quality,
            )

            operations.append(
                {
                    "type": "jpeg",
                    "quality": quality,
                }
            )

        elif operation == "grayscale":
            result = grayscale_to_bgr(result)

            operations.append(
                {
                    "type": "grayscale",
                }
            )

    metadata = {
        "operations": operations,
    }

    return result, metadata


def load_labels(
    labels_path: Path | None,
) -> dict[str, Any]:
    if labels_path is None:
        return {}

    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file does not exist: {labels_path}")

    with labels_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Folder containing original high-quality images.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output dataset directory.",
    )

    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional labels.json file.",
    )

    parser.add_argument(
        "--augmentations-per-image",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    labels = load_labels(args.labels)

    images_output = args.output / "images"
    images_output.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = args.output / "manifest.jsonl"

    source_images = sorted(
        path
        for path in args.input.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )

    if not source_images:
        raise RuntimeError(f"No supported images found in {args.input}")

    with manifest_path.open(
        "w",
        encoding="utf-8",
    ) as manifest:
        for source_path in source_images:
            if labels and source_path.name not in labels:
                print(f"WARNING: No label found for {source_path.name}")

            image = cv2.imread(str(source_path))

            if image is None:
                print(f"WARNING: Could not read {source_path}")
                continue

            source_id = source_path.stem

            # Save original as part of generated dataset.
            original_name = f"{source_id}__original.jpg"

            original_output = images_output / original_name

            cv2.imwrite(
                str(original_output),
                image,
            )

            original_record = {
                "image": f"images/{original_name}",
                "source_image": source_path.name,
                "label_id": source_path.name,
                "variant": "original",
                "augmentation": None,
            }

            manifest.write(json.dumps(original_record) + "\n")

            # Generate augmented images.
            for index in range(
                1,
                args.augmentations_per_image + 1,
            ):
                augmented, metadata = random_augmentation(
                    image,
                    rng,
                    np_rng,
                )

                output_name = f"{source_id}__aug_{index:03d}.jpg"

                output_path = images_output / output_name

                cv2.imwrite(
                    str(output_path),
                    augmented,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        95,
                    ],
                )

                record = {
                    "image": f"images/{output_name}",
                    "source_image": source_path.name,
                    "label_id": source_path.name,
                    "variant": "augmented",
                    "augmentation": metadata,
                }

                manifest.write(json.dumps(record) + "\n")

            print(f"Processed {source_path.name}")

    print()
    print(f"Dataset written to: {args.output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
