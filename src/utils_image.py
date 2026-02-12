from pathlib import Path

import cv2
import numpy as np


def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


def locate_plate_region(image_bgr: np.ndarray) -> np.ndarray:
    """
    Try to detect a likely plate region.
    If no good candidate is found, return original image.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 100, 200)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h, w = gray.shape
    best_box = None
    best_area = 0

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < 0.02 * (w * h):
            continue
        ratio = cw / float(ch + 1e-6)
        if ratio < 2.0 or ratio > 6.5:
            continue
        if area > best_area:
            best_area = area
            best_box = (x, y, cw, ch)

    if best_box is None:
        return image_bgr

    x, y, cw, ch = best_box
    margin_x = max(5, int(0.02 * cw))
    margin_y = max(5, int(0.08 * ch))
    x0 = max(0, x - margin_x)
    y0 = max(0, y - margin_y)
    x1 = min(w, x + cw + margin_x)
    y1 = min(h, y + ch + margin_y)

    return image_bgr[y0:y1, x0:x1]


def preprocess_plate_for_segmentation(plate_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # White characters on black background are easier for contour extraction.
    binary_inv = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        31,
        15,
    )
    kernel = np.ones((2, 2), np.uint8)
    binary_inv = cv2.morphologyEx(binary_inv, cv2.MORPH_OPEN, kernel)
    return binary_inv


def segment_characters(binary_inv_plate: np.ndarray) -> list[tuple[int, int, int, int]]:
    contours, _ = cv2.findContours(
        binary_inv_plate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    h, w = binary_inv_plate.shape
    boxes: list[tuple[int, int, int, int]] = []
    min_area = max(25, int(0.0015 * h * w))

    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        area = cw * ch
        if area < min_area:
            continue
        if ch < 0.25 * h:
            continue
        if cw > 0.6 * w:
            continue
        boxes.append((x, y, cw, ch))

    boxes.sort(key=lambda b: b[0])
    return boxes


def prepare_char_for_model(binary_inv_plate: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    char = binary_inv_plate[y : y + h, x : x + w]

    if char.size == 0:
        raise ValueError("Character crop is empty")

    # Fit largest side to 20 px and center it on a 28x28 canvas.
    target_inner = 20
    scale = target_inner / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(char, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    x = canvas.astype(np.float32) / 255.0
    x = x.reshape(28, 28, 1)
    return x


def draw_boxes(image_bgr: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    out = image_bgr.copy()
    for idx, (x, y, w, h) in enumerate(boxes, start=1):
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            out,
            str(idx),
            (x, max(15, y - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    return out
