"""
Student ID and group extraction from the graphical bubble grid on page 1.

The ID grid is a matrix of filled/empty bubbles.  Each column encodes one digit
in a decimal representation (bubbles 0-9 from top to bottom).  The group grid
follows the same principle with fewer columns.

All detection uses low-level morphological operations — no high-level detectors.
"""

import cv2
import numpy as np
from utils.image_processing import preprocess, morpho_open, detect_grid_cells


# ── Tuneable parameters ────────────────────────────────────────────────────────
# These are calibrated on the reference form; adjust if the form layout changes.

# Relative position of the Student-ID grid inside the first page image
# (x_ratio, y_ratio, w_ratio, h_ratio)  — fractions of page width / height
STUDENT_ID_REGION = (0.05, 0.30, 0.50, 0.18)   # 5-digit ID: 5 cols × 10 rows
STUDENT_ID_DIGITS = 5                            # number of digit columns
STUDENT_ID_ROWS = 10                             # 0-9

GROUP_REGION = (0.60, 0.30, 0.35, 0.18)         # group grid
GROUP_COLS = 4                                   # e.g. G01B  → 4 columns
GROUP_ROWS = 10

SIGNATURE_REGION = (0.05, 0.70, 0.90, 0.25)     # signature zone


def _locate_grid(page_gray, rel_region):
    """Convert relative region coords to absolute pixel coords."""
    h, w = page_gray.shape
    x = int(rel_region[0] * w)
    y = int(rel_region[1] * h)
    bw = int(rel_region[2] * w)
    bh = int(rel_region[3] * h)
    return x, y, bw, bh


def read_bubble_column(grid_bool, col):
    """
    Read a single column of bubbles and return the filled row index (0-9).
    Returns -1 if no bubble or more than one bubble is filled.
    """
    filled = [r for r in range(grid_bool.shape[0]) if grid_bool[r, col]]
    if len(filled) == 1:
        return filled[0]
    return -1


def extract_student_id(page_gray):
    """
    Extract the numeric student ID from the bubble grid on page 1.

    Returns
    -------
    student_id : str  (e.g. "48271") or empty string on failure.
    """
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
    """
    Extract the group code from its bubble grid.

    Returns
    -------
    group : str  (e.g. "G02B") or empty string on failure.
    """
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
    """
    Crop and return the signature sub-image from page 1.
    """
    x, y, w, h = _locate_grid(page_gray, SIGNATURE_REGION)
    return page_gray[y:y+h, x:x+w]
