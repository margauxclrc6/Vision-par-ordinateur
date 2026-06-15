"""
Extracts student ID and group from the bubble grid on page 1.
Each column encodes one digit/letter (bubbles 0-9 top to bottom).

Coordinate system: fractions of image width/height.
The presence photos are camera shots of a printed form, so coordinates
are approximate; the grid detector uses robust contour-based methods.
"""

import cv2
import numpy as np
from utils.image_processing import preprocess, morpho_open, detect_grid_cells


# ── Region definitions (fraction of image width/height) ─────────────────────
# These cover the A4 form as seen in a roughly-centered camera photo.

STUDENT_ID_REGION = (0.73, 0.18, 0.24, 0.38)   # 5-digit ID: 5 cols × 10 rows
STUDENT_ID_DIGITS = 5
STUDENT_ID_ROWS   = 10   # rows 0-9

GROUP_REGION   = (0.35, 0.18, 0.26, 0.38)
GROUP_COLS     = 3        # col0=digit, col1=digit, col2=letter A-J
GROUP_ROWS     = 10
GROUP_LETTER_COL = 2      # column index that encodes a letter (row 0→A … 9→J)

SIGNATURE_REGION = (0.17, 0.23, 0.17, 0.12)  # tightened to signature box only
# ─────────────────────────────────────────────────────────────────────────────


def _locate_grid(page_gray, rel_region):
    """Convert relative region to absolute pixel coordinates."""
    h, w = page_gray.shape
    x  = int(rel_region[0] * w)
    y  = int(rel_region[1] * h)
    bw = int(rel_region[2] * w)
    bh = int(rel_region[3] * h)
    return x, y, bw, bh


def read_bubble_column(grid_bool, col):
    """Return the filled row index (0-9) for a column, or -1 if ambiguous/empty."""
    filled = [r for r in range(grid_bool.shape[0]) if grid_bool[r, col]]
    return filled[0] if len(filled) == 1 else -1


def _grid_to_string(grid, n_cols, letter_col=None):
    """Convert a boolean grid to a string of digits/letters."""
    chars = []
    for col in range(n_cols):
        d = read_bubble_column(grid, col)
        if d < 0:
            chars.append("?")
        elif letter_col is not None and col == letter_col:
            chars.append(chr(ord('A') + d))
        else:
            chars.append(str(d))
    return "".join(chars)


def extract_student_id(page_gray):
    """
    Extract the numeric student ID from the bubble grid.
    Returns e.g. '63807' or a string with '?' for unread columns.
    """
    binary = preprocess(page_gray)
    x, y, w, h = _locate_grid(page_gray, STUDENT_ID_REGION)
    grid = detect_grid_cells(binary, STUDENT_ID_ROWS, STUDENT_ID_DIGITS,
                             region=(x, y, w, h))
    return _grid_to_string(grid, STUDENT_ID_DIGITS)


def extract_group(page_gray):
    """
    Extract the group code (e.g. '78H') from its bubble grid.
    """
    binary = preprocess(page_gray)
    x, y, w, h = _locate_grid(page_gray, GROUP_REGION)
    grid = detect_grid_cells(binary, GROUP_ROWS, GROUP_COLS,
                             region=(x, y, w, h))
    return _grid_to_string(grid, GROUP_COLS, letter_col=GROUP_LETTER_COL)


def extract_signature_region(page_gray):
    """Crop and return the signature sub-image from the presence photo."""
    x, y, w, h = _locate_grid(page_gray, SIGNATURE_REGION)
    return page_gray[y:y + h, x:x + w]
