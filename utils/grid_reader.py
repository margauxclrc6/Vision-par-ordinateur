"""
Extracts student ID and group from the bubble grid on page 1.
Each column encodes one digit (bubbles 0-9 top to bottom).
"""

import cv2
import numpy as np
from utils.image_processing import preprocess, morpho_open, detect_grid_cells


# Relative position of the Student-ID grid inside the first page image
# (x_ratio, y_ratio, w_ratio, h_ratio) — fractions of page width / height
STUDENT_ID_REGION = (0.62, 0.19, 0.35, 0.36)   # 5-digit ID: 5 cols × 10 rows
STUDENT_ID_DIGITS = 5
STUDENT_ID_ROWS = 10                             # 0-9

GROUP_REGION = (0.37, 0.19, 0.24, 0.36)
GROUP_COLS = 3                                   # col1=digit, col2=letter (e.g. 04E)
GROUP_ROWS = 10

SIGNATURE_REGION = (0.03, 0.24, 0.30, 0.26)


def _locate_grid(page_gray, rel_region):
    """Convert relative region coords to absolute pixel coords."""
    h, w = page_gray.shape
    x = int(rel_region[0] * w)
    y = int(rel_region[1] * h)
    bw = int(rel_region[2] * w)
    bh = int(rel_region[3] * h)
    return x, y, bw, bh


def read_bubble_column(grid_bool, col):
    """Return filled row index (0-9) for a column, or -1 if ambiguous."""
    filled = [r for r in range(grid_bool.shape[0]) if grid_bool[r, col]]
    if len(filled) == 1:
        return filled[0]
    return -1


def extract_student_id(page_gray):
    """Extract the numeric student ID from the bubble grid. Returns e.g. '48271' or '' on failure."""
    binary = preprocess(page_gray)
    x, y, w, h = _locate_grid(page_gray, STUDENT_ID_REGION)
    grid = detect_grid_cells(binary, STUDENT_ID_ROWS, STUDENT_ID_DIGITS,
                             region=(x, y, w, h))
    digits = []
    for col in range(STUDENT_ID_DIGITS):
        d = read_bubble_column(grid, col)
        digits.append(str(d) if d >= 0 else "?")
    return "".join(digits)


def extract_group(page_gray):
    """Extract the group code from its bubble grid. Returns e.g. 'G02B' or '' on failure."""
    binary = preprocess(page_gray)
    x, y, w, h = _locate_grid(page_gray, GROUP_REGION)
    grid = detect_grid_cells(binary, GROUP_ROWS, GROUP_COLS,
                             region=(x, y, w, h))
    chars = []
    for col in range(GROUP_COLS):
        d = read_bubble_column(grid, col)
        chars.append(str(d) if d >= 0 else "?")
    return "".join(chars)


def extract_signature_region(page_gray):
    """Crop and return the signature sub-image from page 1."""
    x, y, w, h = _locate_grid(page_gray, SIGNATURE_REGION)
    return page_gray[y:y+h, x:x+w]
