"""
Train a CNN on character data.

Supports two data sources:
  1. EMNIST digit CSV zips  (--train-zip / --test-zip)   → 10 classes (0-9)
  2. Image folder           (--train-dir / --test-dir)    → auto-detected classes

When using image folders the directory layout must be:
    train_dir/
        A/  img1.png  img2.png ...
        B/  ...
        0/  ...
        ...

You can combine both sources with --merge to get a larger training set
that covers digits (from EMNIST) and letters (from plate photos).
"""

import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


DEFAULT_TRAIN_ZIP = "emnist-digits-train.csv.zip"
DEFAULT_TEST_ZIP = "emnist-digits-test.csv.zip"
DEFAULT_MODEL_PATH = "models/digit_model.keras"
LABEL_MAP_PATH = "models/label_map.json"


# ---------------------------------------------------------------------------
# Label mapping helpers
# ---------------------------------------------------------------------------

def build_default_label_map() -> dict[str, int]:
    """0-9 then A-Z → indices 0..35"""
    lm: dict[str, int] = {}
    for i in range(10):
        lm[str(i)] = i
    for i, c in enumerate("ABCDEFGHIJKLMNOPQRSTUVWXYZ"):
        lm[c] = 10 + i
    return lm


def save_label_map(label_map: dict[str, int], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(label_map, f, indent=2)
    print(f"Label map saved to: {path}")


def load_label_map(path: Path) -> dict[str, int]:
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Data loading — EMNIST zips
# ---------------------------------------------------------------------------

def _validate_input_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")


def _load_emnist_zip(zip_path: Path, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    _validate_input_file(zip_path)

    read_kwargs: dict = {"header": None, "compression": "zip"}
    if limit is not None and limit > 0:
        read_kwargs["nrows"] = limit

    df = pd.read_csv(zip_path, **read_kwargs)

    if df.shape[1] != 785:
        raise ValueError(
            f"Unexpected number of columns in {zip_path}. Expected 785, got {df.shape[1]}"
        )

    y = df.iloc[:, 0].to_numpy(dtype=np.int32)
    x = df.iloc[:, 1:].to_numpy(dtype=np.float32)
    return x, y


# ---------------------------------------------------------------------------
# Data loading — image folders
# ---------------------------------------------------------------------------

def _load_image_folder(folder: Path,
                       label_map: dict[str, int],
                       limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Load 28×28 grayscale images from class sub-folders.
    Returns flat pixel arrays (n, 784) and integer labels.
    """
    xs, ys = [], []
    for class_dir in sorted(folder.iterdir()):
        if not class_dir.is_dir():
            continue
        label_str = class_dir.name.upper()
        if label_str not in label_map:
            print(f"  WARN: skipping unknown class folder '{class_dir.name}'")
            continue
        label_int = label_map[label_str]
        imgs = sorted(
            p for p in class_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
        )
        for img_path in imgs:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            if img.shape != (28, 28):
                img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
            xs.append(img.flatten().astype(np.float32))
            ys.append(label_int)
            if limit and len(xs) >= limit:
                break
        if limit and len(xs) >= limit:
            break

    if not xs:
        raise ValueError(f"No valid images found in {folder}")
    return np.array(xs), np.array(ys, dtype=np.int32)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def _preprocess_images(x: np.ndarray) -> np.ndarray:
    x = x / 255.0
    x = x.reshape((-1, 28, 28, 1))
    return x


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    train_zip: Path | None = None,
    test_zip: Path | None = None,
    train_dir: Path | None = None,
    test_dir: Path | None = None,
    model_path: Path = Path(DEFAULT_MODEL_PATH),
    label_map_path: Path = Path(LABEL_MAP_PATH),
    epochs: int = 5,
    batch_size: int = 128,
    train_limit: int | None = None,
    test_limit: int | None = None,
    merge: bool = False,
) -> tuple[float, float]:
    try:
        from src.model import build_model
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing training dependencies. Install requirements and set "
            "KERAS_BACKEND=jax before running training."
        ) from exc

    label_map = build_default_label_map()

    x_parts_train, y_parts_train = [], []
    x_parts_test, y_parts_test = [], []

    # --- EMNIST source ---
    use_emnist = train_zip is not None and Path(train_zip).exists()
    if use_emnist:
        print(f"Loading EMNIST train data from: {train_zip}")
        x_e_train, y_e_train = _load_emnist_zip(Path(train_zip), limit=train_limit)
        x_parts_train.append(x_e_train)
        y_parts_train.append(y_e_train)

        if test_zip and Path(test_zip).exists():
            print(f"Loading EMNIST test data from: {test_zip}")
            x_e_test, y_e_test = _load_emnist_zip(Path(test_zip), limit=test_limit)
            x_parts_test.append(x_e_test)
            y_parts_test.append(y_e_test)

    # --- Image folder source ---
    use_folder = train_dir is not None and Path(train_dir).exists()
    if use_folder:
        print(f"Loading image-folder train data from: {train_dir}")
        x_f_train, y_f_train = _load_image_folder(Path(train_dir), label_map, limit=train_limit)
        x_parts_train.append(x_f_train)
        y_parts_train.append(y_f_train)

        if test_dir and Path(test_dir).exists():
            print(f"Loading image-folder test data from: {test_dir}")
            x_f_test, y_f_test = _load_image_folder(Path(test_dir), label_map, limit=test_limit)
            x_parts_test.append(x_f_test)
            y_parts_test.append(y_f_test)

    if not x_parts_train:
        raise ValueError(
            "No training data found. Provide --train-zip and/or --train-dir."
        )

    # Concatenate all sources
    x_train = np.concatenate(x_parts_train, axis=0)
    y_train = np.concatenate(y_parts_train, axis=0)

    if x_parts_test:
        x_test = np.concatenate(x_parts_test, axis=0)
        y_test = np.concatenate(y_parts_test, axis=0)
    else:
        # Auto split 80/20
        print("No test set provided — using 20% of training data for validation.")
        n = len(x_train)
        perm = np.random.default_rng(42).permutation(n)
        split = int(0.8 * n)
        x_test = x_train[perm[split:]]
        y_test = y_train[perm[split:]]
        x_train = x_train[perm[:split]]
        y_train = y_train[perm[:split]]

    x_train = _preprocess_images(x_train)
    x_test = _preprocess_images(x_test)

    # Determine num_classes from actual data
    num_classes = int(max(y_train.max(), y_test.max())) + 1
    # But always at least 36 if we use the full label map
    num_classes = max(num_classes, 36) if use_folder else max(num_classes, 10)

    print(f"Classes: {num_classes}  |  Train samples: {len(x_train)}  |  Test samples: {len(x_test)}")

    model = build_model(num_classes=num_classes)
    model.summary()

    print(f"Training for {epochs} epoch(s) ...")
    model.fit(
        x_train,
        y_train,
        validation_data=(x_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=2,
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path)
    print(f"Model saved to: {model_path}")

    save_label_map(label_map, label_map_path)

    return float(test_loss), float(test_acc)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train CNN on EMNIST CSVs and/or image folders"
    )
    # EMNIST source
    parser.add_argument("--train-zip", default=DEFAULT_TRAIN_ZIP,
                        help="Path to EMNIST train CSV zip")
    parser.add_argument("--test-zip", default=DEFAULT_TEST_ZIP,
                        help="Path to EMNIST test CSV zip")
    # Image folder source
    parser.add_argument("--train-dir", default=None,
                        help="Path to image folder with class sub-dirs for training")
    parser.add_argument("--test-dir", default=None,
                        help="Path to image folder with class sub-dirs for testing")
    # Merge
    parser.add_argument("--merge", action="store_true",
                        help="Merge EMNIST and image-folder data for training")
    # Output
    parser.add_argument("--model-out", default=DEFAULT_MODEL_PATH, help="Output model path")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--train-limit", type=int, default=None,
                        help="Optional row limit for training data (smoke tests)")
    parser.add_argument("--test-limit", type=int, default=None,
                        help="Optional row limit for test data (smoke tests)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Decide which sources to pass
    train_zip = Path(args.train_zip) if args.train_zip else None
    test_zip = Path(args.test_zip) if args.test_zip else None
    train_dir = Path(args.train_dir) if args.train_dir else None
    test_dir = Path(args.test_dir) if args.test_dir else None

    # If --train-dir is given without --merge, don't auto-load EMNIST
    if train_dir and not args.merge:
        train_zip = None
        test_zip = None

    train_model(
        train_zip=train_zip,
        test_zip=test_zip,
        train_dir=train_dir,
        test_dir=test_dir,
        model_path=Path(args.model_out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        merge=args.merge,
    )


if __name__ == "__main__":
    main()
