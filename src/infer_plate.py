import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
from pathlib import Path

import numpy as np

DEFAULT_MODEL_PATH = "models/digit_model.keras"
INDEX_TO_CHAR = {i: str(i) for i in range(10)}


def _collect_images(image: str | None, folder: str | None) -> list[Path]:
    if bool(image) == bool(folder):
        raise ValueError("Provide exactly one of --image or --folder")

    if image:
        p = Path(image)
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Image not found: {p}")
        return [p]

    folder_path = Path(folder)  # type: ignore[arg-type]
    if not folder_path.exists() or not folder_path.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    images = sorted([p for p in folder_path.iterdir() if p.suffix.lower() in exts])
    if not images:
        raise ValueError(f"No images found in folder: {folder_path}")
    return images


def predict_plate_text(
    model,
    image_path: Path,
    assume_cropped_plate: bool = False,
    debug_out_dir: Path | None = None,
) -> str:
    from src.utils_image import (
        draw_boxes,
        load_image,
        locate_plate_region,
        prepare_char_for_model,
        preprocess_plate_for_segmentation,
        segment_characters,
    )

    image = load_image(image_path)
    plate_region = image if assume_cropped_plate else locate_plate_region(image)

    binary_inv = preprocess_plate_for_segmentation(plate_region)
    boxes = segment_characters(binary_inv)

    if not boxes:
        raise RuntimeError(
            f"No characters segmented from image: {image_path}. "
            "Try --assume-cropped-plate if input is already a close crop."
        )

    char_tensors = [prepare_char_for_model(binary_inv, b) for b in boxes]
    x = np.stack(char_tensors, axis=0)

    preds = model.predict(x, verbose=0)
    indices = np.argmax(preds, axis=1).tolist()
    text = "".join(INDEX_TO_CHAR.get(i, "?") for i in indices)

    if debug_out_dir is not None:
        import cv2

        debug_out_dir.mkdir(parents=True, exist_ok=True)
        debug_vis = draw_boxes(plate_region, boxes)
        out_path = debug_out_dir / f"{image_path.stem}_debug.png"
        cv2.imwrite(str(out_path), debug_vis)

    return text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Infer license plate text from image(s)")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="Path to trained .keras model")
    parser.add_argument("--image", default=None, help="Path to one input image")
    parser.add_argument("--folder", default=None, help="Path to folder of images")
    parser.add_argument(
        "--assume-cropped-plate",
        action="store_true",
        help="Skip plate detection and treat the full input image as plate region",
    )
    parser.add_argument(
        "--debug-out",
        default=None,
        help="Optional folder to save debug visualization with bounding boxes",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        from keras.models import load_model
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing inference dependencies. Install requirements and set "
            "KERAS_BACKEND=jax before running inference."
        ) from exc

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Run training first with 'python main.py train'."
        )

    images = _collect_images(args.image, args.folder)
    model = load_model(model_path)

    debug_dir = Path(args.debug_out) if args.debug_out else None

    for image_path in images:
        try:
            pred = predict_plate_text(
                model=model,
                image_path=image_path,
                assume_cropped_plate=args.assume_cropped_plate,
                debug_out_dir=debug_dir,
            )
            print(f"{image_path}: {pred}")
        except Exception as exc:  # noqa: BLE001
            print(f"{image_path}: ERROR - {exc}")


if __name__ == "__main__":
    main()
