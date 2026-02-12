import os

os.environ.setdefault("KERAS_BACKEND", "jax")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TRAIN_ZIP = "emnist-digits-train.csv.zip"
DEFAULT_TEST_ZIP = "emnist-digits-test.csv.zip"
DEFAULT_MODEL_PATH = "models/digit_model.keras"


def _validate_input_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Input path is not a file: {path}")


def _load_emnist_zip(zip_path: Path, limit: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    _validate_input_file(zip_path)

    read_kwargs = {"header": None, "compression": "zip"}
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


def _preprocess_images(x: np.ndarray) -> np.ndarray:
    x = x / 255.0
    x = x.reshape((-1, 28, 28, 1))
    return x


def train_model(
    train_zip: Path,
    test_zip: Path,
    model_path: Path,
    epochs: int = 5,
    batch_size: int = 128,
    train_limit: int | None = None,
    test_limit: int | None = None,
) -> tuple[float, float]:
    try:
        from src.model import build_model
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing training dependencies. Install requirements and set "
            "KERAS_BACKEND=jax before running training."
        ) from exc

    print(f"Loading train data from: {train_zip}")
    x_train, y_train = _load_emnist_zip(train_zip, limit=train_limit)
    print(f"Loading test data from: {test_zip}")
    x_test, y_test = _load_emnist_zip(test_zip, limit=test_limit)

    x_train = _preprocess_images(x_train)
    x_test = _preprocess_images(x_test)

    model = build_model(num_classes=10)
    model.summary()

    print(f"Training with {len(x_train)} samples for {epochs} epoch(s)...")
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

    return float(test_loss), float(test_acc)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CNN on EMNIST digits CSV zip files")
    parser.add_argument("--train-zip", default=DEFAULT_TRAIN_ZIP, help="Path to train CSV zip")
    parser.add_argument("--test-zip", default=DEFAULT_TEST_ZIP, help="Path to test CSV zip")
    parser.add_argument("--model-out", default=DEFAULT_MODEL_PATH, help="Output model path")
    parser.add_argument("--epochs", type=int, default=5, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument(
        "--train-limit",
        type=int,
        default=None,
        help="Optional row limit for training set (smoke tests)",
    )
    parser.add_argument(
        "--test-limit",
        type=int,
        default=None,
        help="Optional row limit for test set (smoke tests)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_model(
        train_zip=Path(args.train_zip),
        test_zip=Path(args.test_zip),
        model_path=Path(args.model_out),
        epochs=args.epochs,
        batch_size=args.batch_size,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
    )


if __name__ == "__main__":
    main()
