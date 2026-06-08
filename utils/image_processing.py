"""Low-level image processing utilities (filtering, Hough, morphology, rotation)."""

import cv2
import numpy as np


def load_image(path):
    """Load image as grayscale and color."""
    img_color = cv2.imread(str(path))
    if img_color is None:
        raise FileNotFoundError(f"Cannot load image: {path}")
    img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
    return img_color, img_gray


def preprocess(gray):
    """Denoise and binarize a grayscale image."""
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def deskew(gray):
    """Correct document skew using Hough lines. Returns (deskewed_image, angle_degrees)."""
    binary = preprocess(gray)
    edges = cv2.Canny(binary, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=200)

    angle = 0.0
    if lines is not None:
        angles = []
        for rho, theta in lines[:, 0]:
            a = np.degrees(theta) - 90
            if abs(a) < 45:
                angles.append(a)
        if angles:
            angle = np.median(angles)

    h, w = gray.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    deskewed = cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)
    return deskewed, angle


def morpho_open(binary, ksize=3):
    """Morphological opening to remove small noise."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)


def morpho_close(binary, ksize=5):
    """Morphological closing to fill small holes."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def find_horizontal_lines(binary, min_len_ratio=0.3):
    """Detect horizontal lines via morphological erosion. Returns a binary mask."""
    h, w = binary.shape
    min_len = int(w * min_len_ratio)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_len, 1))
    inv = cv2.bitwise_not(binary)
    eroded = cv2.erode(inv, kernel)
    dilated = cv2.dilate(eroded, kernel)
    return dilated


def find_vertical_lines(binary, min_len_ratio=0.3):
    """Detect vertical lines via morphological erosion. Returns a binary mask."""
    h, w = binary.shape
    min_len = int(h * min_len_ratio)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, min_len))
    inv = cv2.bitwise_not(binary)
    eroded = cv2.erode(inv, kernel)
    dilated = cv2.dilate(eroded, kernel)
    return dilated


def detect_grid_cells(binary, n_rows, n_cols, region=None):
    """
    Detect filled bubble cells using contour-based grid detection with k-means
    clustering of bubble positions. Falls back to equal-grid division if not
    enough candidates are found.
    """
    if region is not None:
        x0, y0, rw, rh = region
        roi = binary[y0:y0+rh, x0:x0+rw]
    else:
        roi = binary

    roi_h, roi_w = roi.shape
    inv = cv2.bitwise_not(roi)

    exp_h = roi_h / n_rows
    exp_w = roi_w / n_cols

    # Find contours of bubble-sized square objects
    cnts, _ = cv2.findContours(inv.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    seen = set()
    candidates = []
    for cnt in cnts:
        bx, by, bw, bh_ = cv2.boundingRect(cnt)
        if not (exp_w * 0.25 < bw < exp_w * 2.2 and
                exp_h * 0.25 < bh_ < exp_h * 2.2):
            continue
        if not (0.4 < bw / max(bh_, 1) < 2.5):
            continue
        key = (bx // 15, by // 15)
        if key in seen:
            continue
        seen.add(key)
        cx, cy = bx + bw // 2, by + bh_ // 2
        pad = max(2, int(min(bw, bh_) * 0.15))
        inner = inv[by + pad: by + bh_ - pad, bx + pad: bx + bw - pad]
        fill = np.sum(inner > 0) / max(inner.size, 1)
        candidates.append((cx, cy, fill))

    grid = np.zeros((n_rows, n_cols), dtype=bool)

    if len(candidates) < max(n_rows, n_cols):
        # Fallback: equal-grid division
        cell_h = roi_h // n_rows
        cell_w = roi_w // n_cols
        for r in range(n_rows):
            for c in range(n_cols):
                cell = inv[r * cell_h:(r + 1) * cell_h, c * cell_w:(c + 1) * cell_w]
                grid[r, c] = np.sum(cell > 0) / max(cell.size, 1) > 0.15
        return grid

    pts_x = np.array([[c[0]] for c in candidates], dtype=np.float32)
    pts_y = np.array([[c[1]] for c in candidates], dtype=np.float32)

    n_c = min(n_cols, len(candidates))
    n_r = min(n_rows, len(candidates))

    _, col_lbls, col_centers = cv2.kmeans(
        pts_x, n_c, None,
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0),
        10, cv2.KMEANS_PP_CENTERS)
    _, row_lbls, row_centers = cv2.kmeans(
        pts_y, n_r, None,
        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER, 50, 1.0),
        10, cv2.KMEANS_PP_CENTERS)

    col_sorted = np.argsort(col_centers.flatten())
    row_sorted = np.argsort(row_centers.flatten())
    col_rank = np.empty(n_c, dtype=int)
    col_rank[col_sorted] = np.arange(n_c)
    row_rank = np.empty(n_r, dtype=int)
    row_rank[row_sorted] = np.arange(n_r)

    FILL_THRESH = 0.08
    for i, (cx, cy, fill) in enumerate(candidates):
        c = int(col_rank[col_lbls[i, 0]])
        r = int(row_rank[row_lbls[i, 0]])
        if 0 <= r < n_rows and 0 <= c < n_cols and fill > FILL_THRESH:
            grid[r, c] = True

    return grid


def crop_region(img, x, y, w, h):
    """Crop a rectangular region from an image."""
    return img[y:y+h, x:x+w]


def normalize_signature(sig_gray, target_size=(128, 64)):
    """Binarize, crop to bounding box, then resize to target_size."""
    _, binary = cv2.threshold(sig_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros(target_size[::-1], dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary[y:y+h, x:x+w]
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    return resized


def image_similarity(img_a, img_b):
    """Normalized cross-correlation between two binary images. Returns score in [0, 1]."""
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]),
                           interpolation=cv2.INTER_AREA)
    a = img_a.astype(np.float32) / 255.0
    b = img_b.astype(np.float32) / 255.0
    num = np.sum(a * b)
    den = np.sqrt(np.sum(a**2) * np.sum(b**2))
    return float(num / den) if den > 1e-8 else 0.0
