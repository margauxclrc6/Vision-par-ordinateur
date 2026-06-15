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

GROUP_DIGITS_REGION = (0.50, 0.18, 0.09, 0.38)   # 2 digit columns (e.g. 7,8)
GROUP_LETTER_REGION = (0.615, 0.18, 0.06, 0.38)  # 1 letter column (A-J)
GROUP_ROWS     = 10

SIGNATURE_REGION = (0.02, 0.17, 0.85, 0.42)  # search area containing signature box
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
    The grid has 2 digit columns and a separate letter column (A-J), with an
    unequal gap between them, so each part is read from its own region.
    """
    binary = preprocess(page_gray)

    # Two digit columns
    xd, yd, wd, hd = _locate_grid(page_gray, GROUP_DIGITS_REGION)
    digit_grid = detect_grid_cells(binary, GROUP_ROWS, 2, region=(xd, yd, wd, hd))
    digits = _grid_to_string(digit_grid, 2)

    # One letter column
    xl, yl, wl, hl = _locate_grid(page_gray, GROUP_LETTER_REGION)
    letter_grid = detect_grid_cells(binary, GROUP_ROWS, 1, region=(xl, yl, wl, hl))
    letter = _grid_to_string(letter_grid, 1, letter_col=0)

    return digits + letter


def extract_signature_region(page_gray):
    """
    Find the signature box rectangle in the search area and return its interior.
    Falls back to the full search area if no rectangle is found.
    """
    ph, pw = page_gray.shape
    x, y, w, h = _locate_grid(page_gray, SIGNATURE_REGION)
    roi = page_gray[y:y + h, x:x + w]

    # Threshold and find contours of large rectangles
    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = (w * h) * 0.05
    max_area = (w * h) * 0.80
    best_area = 0
    best_box  = None

    for cnt in cnts:
        area = cv2.contourArea(cnt)
        if not (min_area < area < max_area):
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
        if len(approx) != 4:
            continue
        bx, by, bw, bh = cv2.boundingRect(approx)
        aspect = bw / max(bh, 1)
        if not (0.8 < aspect < 4.0):
            continue
        if area > best_area:
            best_area = area
            best_box  = (bx, by, bw, bh)

    if best_box is not None:
        bx, by, bw, bh = best_box
        pad = 4
        interior = roi[max(0, by + pad): by + bh - pad,
                       max(0, bx + pad): bx + bw - pad]
        if interior.size > 0:
            return interior

    return roi
