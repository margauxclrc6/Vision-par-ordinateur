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
    Divide a region into an n_rows x n_cols grid and return a bool array
    indicating which cells are filled. Uses per-column argmax to handle
    variable bubble fill intensity across different scans.
    """
    if region is not None:
        x0, y0, rw, rh = region
        roi = binary[y0:y0+rh, x0:x0+rw]
    else:
        roi = binary

    roi_h, roi_w = roi.shape
    inv = cv2.bitwise_not(roi)

    cell_h = roi_h / n_rows
    cell_w = roi_w / n_cols

    fill_scores = np.zeros((n_rows, n_cols), dtype=float)
    for r in range(n_rows):
        for c in range(n_cols):
            r0, r1 = int(r * cell_h), int((r + 1) * cell_h)
            c0, c1 = int(c * cell_w), int((c + 1) * cell_w)
            pad_y = max(1, (r1 - r0) // 6)
            pad_x = max(1, (c1 - c0) // 6)
            cell = inv[r0 + pad_y: r1 - pad_y, c0 + pad_x: c1 - pad_x]
            fill_scores[r, c] = np.sum(cell > 0) / max(cell.size, 1)

    grid = np.zeros((n_rows, n_cols), dtype=bool)
    for c in range(n_cols):
        col = fill_scores[:, c]
        max_idx = int(np.argmax(col))
        max_val = col[max_idx]
        med_val = np.median(col)
        # Mark filled if the maximum stands out from the background
        if max_val > 0.03 and max_val > med_val * 1.4:
            grid[max_idx, c] = True

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
