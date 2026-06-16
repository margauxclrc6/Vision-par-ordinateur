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
