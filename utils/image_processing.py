"""Low-level image processing utilities (filtering, Hough, morphology, rotation)."""

import cv2
import numpy as np


def load_image(path):
    """Load image as grayscale and color. Handles HEIC, uppercase extensions, etc."""
    path = str(path)

    # 1. Standard cv2 load
    img_color = cv2.imread(path)

    # 2. np.fromfile fallback (handles uppercase extensions on Linux)
    if img_color is None:
        try:
            raw = np.fromfile(path, dtype=np.uint8)
            img_color = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        except Exception:
            img_color = None

    # 3. Pillow fallback — covers HEIC/HEIF (pillow-heif), WebP, TIFF, etc.
    if img_color is None:
        try:
            from PIL import Image as _PILImage
            # Register HEIC support if pillow-heif is available
            try:
                import pillow_heif
                pillow_heif.register_heif_opener()
            except ImportError:
                pass
            pil_img = _PILImage.open(path).convert("RGB")
            img_color = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        except Exception:
            img_color = None

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
    Detect filled bubble cells. Primary: contour detection + k-means clustering.
    Fallback: equal-grid division with per-column argmax.
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

    # Find square-like bubble contours, strict size filter to exclude label text
    cnts, _ = cv2.findContours(inv.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    seen = set()
    candidates = []
    for cnt in cnts:
        bx, by, bw, bh_ = cv2.boundingRect(cnt)
        if not (exp_w * 0.35 < bw < exp_w * 1.8 and
                exp_h * 0.35 < bh_ < exp_h * 1.8):
            continue
        if not (0.5 < bw / max(bh_, 1) < 2.0):
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

    if len(candidates) < n_cols:
        # Fallback: equal-grid with per-column argmax
        cell_h = roi_h / n_rows
        cell_w = roi_w / n_cols
        fill_scores = np.zeros((n_rows, n_cols), dtype=float)
        for r in range(n_rows):
            for c in range(n_cols):
                r0, r1 = int(r * cell_h), int((r + 1) * cell_h)
                c0, c1 = int(c * cell_w), int((c + 1) * cell_w)
                py = max(1, (r1 - r0) // 6)
                px = max(1, (c1 - c0) // 6)
                cell = inv[r0 + py: r1 - py, c0 + px: c1 - px]
                fill_scores[r, c] = np.sum(cell > 0) / max(cell.size, 1)
        for c in range(n_cols):
            col = fill_scores[:, c]
            idx = int(np.argmax(col))
            if col[idx] > 0.03 and col[idx] > np.median(col) * 1.4:
                grid[idx, c] = True
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

    col_rank = np.empty(n_c, dtype=int)
    col_rank[np.argsort(col_centers.flatten())] = np.arange(n_c)
    row_rank = np.empty(n_r, dtype=int)
    row_rank[np.argsort(row_centers.flatten())] = np.arange(n_r)

    fill_scores = np.zeros((n_rows, n_cols), dtype=float)
    for i, (cx, cy, fill) in enumerate(candidates):
        c = int(col_rank[col_lbls[i, 0]])
        r = int(row_rank[row_lbls[i, 0]])
        if 0 <= r < n_rows and 0 <= c < n_cols:
            fill_scores[r, c] = max(fill_scores[r, c], fill)

    for c in range(n_cols):
        col = fill_scores[:, c]
        idx = int(np.argmax(col))
        if col[idx] > 0.04 and col[idx] > np.median(col) * 1.5:
            grid[idx, c] = True

    return grid


def crop_region(img, x, y, w, h):
    """Crop a rectangular region from an image."""
    return img[y:y+h, x:x+w]


def normalize_signature(sig_gray, target_size=(128, 64)):
    """
    Normalize a signature image for matching.
    Uses adaptive thresholding (robust to camera lighting) + tight crop + resize.
    """
    # Adaptive threshold handles uneven illumination from phone photos
    binary = cv2.adaptiveThreshold(
        sig_gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
        blockSize=25, C=10)
    # Remove salt & pepper noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    coords = cv2.findNonZero(binary)
    if coords is None:
        return np.zeros((target_size[1], target_size[0]), dtype=np.uint8)
    x, y, w, h = cv2.boundingRect(coords)
    pad = 4
    x, y = max(0, x - pad), max(0, y - pad)
    w = min(binary.shape[1] - x, w + 2 * pad)
    h = min(binary.shape[0] - y, h + 2 * pad)
    cropped = binary[y:y + h, x:x + w]
    return cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)


def image_similarity(img_a, img_b):
    """Cosine similarity between two binary images. Returns score in [0, 1].
    Used for cryptogram comparison."""
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]),
                           interpolation=cv2.INTER_AREA)
    a = img_a.astype(np.float32) / 255.0
    b = img_b.astype(np.float32) / 255.0
    num = np.sum(a * b)
    den = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    return float(num / den) if den > 1e-8 else 0.0


def ncc_similarity(img_a, img_b):
    """Zero-mean NCC between two images. Returns score in [-1, 1].
    Used for signature matching."""
    if img_a.shape != img_b.shape:
        img_b = cv2.resize(img_b, (img_a.shape[1], img_a.shape[0]),
                           interpolation=cv2.INTER_AREA)
    a = img_a.astype(np.float32)
    b = img_b.astype(np.float32)
    a -= a.mean()
    b -= b.mean()
    num = np.sum(a * b)
    den = np.sqrt(np.sum(a ** 2) * np.sum(b ** 2))
    return float(num / den) if den > 1e-8 else 0.0
