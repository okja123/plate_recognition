from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Failed to read image: {path}")
    return image


# ---------------------------------------------------------------------------
# Plate localisation (heuristic, contour-based)
# ---------------------------------------------------------------------------

def locate_plate_region(image_bgr: np.ndarray) -> np.ndarray:
    """
    Try to detect a likely plate region via edge-based contour search.
    Falls back to the full image when nothing plausible is found.
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


# ---------------------------------------------------------------------------
# Perspective correction  (ported from images_treatment.py)
# ---------------------------------------------------------------------------

def _order_corner_points(pts: np.ndarray) -> np.ndarray:
    """Order four points as: top-left, top-right, bottom-left, bottom-right."""
    # Sort by y first, then by x within each pair
    sorted_by_y = pts[pts[:, 1].argsort()]
    top = sorted_by_y[:2]
    bottom = sorted_by_y[2:]
    top = top[top[:, 0].argsort()]
    bottom = bottom[bottom[:, 0].argsort()]
    return np.array([top[0], top[1], bottom[0], bottom[1]], dtype=np.float32)


def _detect_plate_corners(contour: np.ndarray, image_shape: tuple) -> np.ndarray | None:
    """
    Given the main plate contour, estimate four corner points using
    the curvature-score method from images_treatment.py.
    Returns ordered 4×2 float32 array or None.
    """
    contour = np.squeeze(contour, axis=1) if contour.ndim == 3 else contour
    n = len(contour)
    if n < 10:
        return None

    range_point = max(5, n // 8)
    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        p_prev = contour[(i - range_point) % n].astype(np.float64)
        p_next = contour[(i + range_point) % n].astype(np.float64)
        midpoint = (p_prev + p_next) / 2.0
        scores[i] = np.linalg.norm(contour[i].astype(np.float64) - midpoint)

    if scores.max() == 0:
        return None

    scores = (scores / scores.max() * 255).astype(np.uint8)
    threshold = 150
    indices = np.where(scores > threshold)[0]
    if len(indices) < 4:
        return None

    # Cluster high-score points into 4 corners via k-means-style grouping
    # with image-corner seeds
    h_img, w_img = image_shape[:2]
    seeds = np.array([
        [0, 0],
        [w_img, 0],
        [0, h_img],
        [w_img, h_img],
    ], dtype=np.float64)

    candidate_pts = contour[indices].astype(np.float64)
    candidate_scores = scores[indices].astype(np.float64)

    corners = []
    for seed in seeds:
        dists = np.linalg.norm(candidate_pts - seed, axis=1)
        # Weight: prefer high score AND close to expected corner
        weights = candidate_scores / (dists + 1e-6)
        best_idx = np.argmax(weights)
        corners.append(candidate_pts[best_idx])

    corners = np.array(corners, dtype=np.float32)

    # Sanity: all four corners should be distinct
    if len(set(map(tuple, corners.tolist()))) < 4:
        return None

    return _order_corner_points(corners)


def correct_perspective(image_bgr: np.ndarray,
                        target_width: int = 400,
                        target_height: int = 120) -> np.ndarray:
    """
    Detect the plate quadrilateral and warp to a rectangular view.
    Returns the original image when detection fails.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((5, 5), np.float32) / 25
    blurred = cv2.filter2D(gray, -1, kernel)
    _, binary = cv2.threshold(blurred, int(np.mean(gray)), 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return image_bgr

    main_contour = max(contours, key=cv2.contourArea)

    # Try polygon approximation first (faster, more robust for clean plates)
    peri = cv2.arcLength(main_contour, True)
    approx = cv2.approxPolyDP(main_contour, 0.02 * peri, True)
    if len(approx) == 4:
        src_pts = _order_corner_points(approx.reshape(4, 2).astype(np.float32))
    else:
        src_pts = _detect_plate_corners(main_contour, gray.shape)

    if src_pts is None:
        return image_bgr

    dst_pts = np.array([
        [0, 0],
        [target_width, 0],
        [0, target_height],
        [target_width, target_height],
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(image_bgr, M, (target_width, target_height))
    return warped


# ---------------------------------------------------------------------------
# Plate preprocessing & character segmentation
# ---------------------------------------------------------------------------

def preprocess_plate_for_segmentation(plate_bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(plate_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

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
        # Too short → noise / small dots
        if ch < 0.35 * h:
            continue
        # Too wide → plate border
        if cw > 0.6 * w:
            continue
        # In the leftmost 12% of the plate → EU flag / country strip
        if x + cw / 2 < 0.12 * w:
            continue
        # Very thin vertically → dash separators (aspect ratio width/height)
        if cw / max(ch, 1) < 0.25:
            continue
        boxes.append((x, y, cw, ch))

    boxes.sort(key=lambda b: b[0])
    return boxes


# ---------------------------------------------------------------------------
# Character image preparation (28×28 canvas)
# ---------------------------------------------------------------------------

def prepare_char_for_model(binary_inv_plate: np.ndarray,
                           box: tuple[int, int, int, int]) -> np.ndarray:
    x, y, w, h = box
    char = binary_inv_plate[y : y + h, x : x + w]

    if char.size == 0:
        raise ValueError("Character crop is empty")

    target_inner = 20
    scale = target_inner / float(max(h, w))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    resized = cv2.resize(char, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_offset = (28 - new_w) // 2
    y_offset = (28 - new_h) // 2
    canvas[y_offset : y_offset + new_h, x_offset : x_offset + new_w] = resized

    out = canvas.astype(np.float32) / 255.0
    out = out.reshape(28, 28, 1)
    return out


# ---------------------------------------------------------------------------
# Debug visualisation
# ---------------------------------------------------------------------------

def draw_boxes(image_bgr: np.ndarray,
               boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
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
