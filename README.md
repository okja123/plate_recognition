# Number Plate Recognition Baseline (Pandas + Keras/JAX)

This repository provides a complete baseline focused on **character recognition** for license plates:

1. Train a CNN on EMNIST digits from zipped CSV files with **pandas**.
2. Save model to `models/digit_model.keras`.
3. Run inference on plate images (`--image` or `--folder`) with simple plate localization + character segmentation.

## Project Structure

```text
plate_recognition/
├── emnist-digits-train.csv.zip
├── emnist-digits-test.csv.zip
├── main.py
├── requirements.txt
├── src/
│   ├── model.py
│   ├── train_emnist.py
│   ├── infer_plate.py
│   └── utils_image.py
└── models/
    └── digit_model.keras (generated)
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
export KERAS_BACKEND=jax
```

## Train

Full training:

```bash
python main.py train --epochs 5 --batch-size 128
```

Model output:

```text
models/digit_model.keras
```

### Smoke Test (fast)

Use a subset to confirm the pipeline runs:

```bash
python main.py train --epochs 1 --train-limit 5000 --test-limit 1000 --batch-size 64
```

## Inference

Single image:

```bash
python main.py infer --image path/to/plate.jpg --model models/digit_model.keras
```

Folder of images:

```bash
python main.py infer --folder path/to/images --model models/digit_model.keras
```

If images are already tightly cropped to only the plate region:

```bash
python main.py infer --folder path/to/images --assume-cropped-plate
```

Save debug images with character boxes:

```bash
python main.py infer --folder path/to/images --debug-out outputs/debug
```

## Notes

- EMNIST digits classes are `0-9`, so predictions are digits only in this baseline.
- Plate detection is heuristic-based; if detection is unstable, use `--assume-cropped-plate`.
- Errors are reported per image (missing file, no segmented characters, unreadable image, etc.).
