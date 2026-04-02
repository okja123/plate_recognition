"""
Prepare training data from license plate photos.

Naming convention: each image file should be named with the plate text
(stripping dashes/spaces). Examples:
    AB123CD.png   ->  characters A, B, 1, 2, 3, C, D
    AB-123-CD.jpg ->  same (dashes are stripped)

The script segments characters from each plate photo and saves individual
28×28 grayscale character images into class sub-folders:
    <output_dir>/A/  <output_dir>/B/  <output_dir>/1/  ...

This folder structure is compatible with the image-folder training mode.
"""

import argparse
import re
from pathlib import Path

import cv2
import numpy as np

from src.utils_image import (
    correct_perspective,
    load_image,
    locate_plate_region,
    preprocess_plate_for_segmentation,
    segment_characters,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _clean_plate_text(filename_stem: str) -> str:
    """Remove dashes, spaces, underscores — keep only alphanumeric chars."""
    return re.sub(r"[^A-Za-z0-9]", "", filename_stem).upper()


def _save_char_image(canvas_28: np.ndarray, label: str, output_dir: Path, index: int) -> Path:
    """Save a 28×28 uint8 image into output_dir/<label>/."""
    label_dir = output_dir / label
    label_dir.mkdir(parents=True, exist_ok=True)
    # Use a counter to avoid collisions
    out_path = label_dir / f"{label}_{index:06d}.png"
    cv2.imwrite(str(out_path), canvas_28)
    return out_path


def _extract_char_canvas(binary_inv: np.ndarray,
                         box: tuple[int, int, int, int]) -> np.ndarray:
    """Extract a character crop and center it on a 28×28 canvas (uint8)."""
    x, y, w, h = box
    char = binary_inv[y : y + h, x : x + w]
    if char.size == 0:
        raise ValueError("Empty character crop")

    target_inner = 20
    scale = target_inner / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(char, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    canvas[y_off : y_off + new_h, x_off : x_off + new_w] = resized
    return canvas


def prepare_from_folder(
    input_folder: Path,
    output_dir: Path,
    assume_cropped_plate: bool = False,
    use_perspective: bool = True,
) -> dict[str, int]:
    """
    Process all plate images in *input_folder* and write labelled character
    images to *output_dir*.

    Returns a dict  {label: count}  summarising what was extracted.
    """
    images = sorted(
        p for p in input_folder.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )
    if not images:
        raise ValueError(f"No images found in {input_folder}")

    global_counter = 0
    stats: dict[str, int] = {}

    for img_path in images:
        plate_text = _clean_plate_text(img_path.stem)
        if not plate_text:
            print(f"  SKIP {img_path.name} — cannot derive plate text from filename")
            continue

        try:
            image = load_image(img_path)
        except ValueError as exc:
            print(f"  SKIP {img_path.name} — {exc}")
            continue

        # Plate region extraction
        if assume_cropped_plate:
            plate = image
        else:
            plate = locate_plate_region(image)

        # Perspective correction (optional)
        if use_perspective:
            plate = correct_perspective(plate)

        binary_inv = preprocess_plate_for_segmentation(plate)
        boxes = segment_characters(binary_inv)

        if len(boxes) != len(plate_text):
            print(
                f"  WARN {img_path.name} — segmented {len(boxes)} chars "
                f"but plate text has {len(plate_text)} chars ('{plate_text}'). "
                f"Skipping this image."
            )
            continue

        for char_label, box in zip(plate_text, boxes):
            try:
                canvas = _extract_char_canvas(binary_inv, box)
                _save_char_image(canvas, char_label, output_dir, global_counter)
                global_counter += 1
                stats[char_label] = stats.get(char_label, 0) + 1
            except ValueError as exc:
                print(f"  WARN {img_path.name} char '{char_label}' — {exc}")

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare labelled character images from plate photos"
    )
    parser.add_argument(
        "--input", required=True,
        help="Folder containing plate images (named with plate text)",
    )
    parser.add_argument(
        "--output", default="data/chars",
        help="Output folder for labelled character images (default: data/chars)",
    )
    parser.add_argument(
        "--assume-cropped-plate", action="store_true",
        help="Skip plate detection (images are already tightly cropped)",
    )
    parser.add_argument(
        "--no-perspective", action="store_true",
        help="Skip perspective correction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_folder = Path(args.input)
    output_dir = Path(args.output)

    print(f"Input folder : {input_folder}")
    print(f"Output folder: {output_dir}")

    stats = prepare_from_folder(
        input_folder=input_folder,
        output_dir=output_dir,
        assume_cropped_plate=args.assume_cropped_plate,
        use_perspective=not args.no_perspective,
    )

    total = sum(stats.values())
    print(f"\nDone — extracted {total} character images across {len(stats)} classes:")
    for label in sorted(stats):
        print(f"  {label}: {stats[label]}")


if __name__ == "__main__":
    main()
