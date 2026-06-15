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

# Group grid: header row ("7 8 H") excluded, bubble rows 0-9 start at y≈0.215
GROUP_DIGITS_REGION = (0.511, 0.215, 0.060, 0.283)  # 2 digit columns (e.g. 7,8)
GROUP_LETTER_REGION = (0.622, 0.215, 0.045, 0.283)  # 1 letter column (A-J)
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


def _read_fixed_grid(binary, region, n_rows, n_cols):
    """
    Read a bubble grid by dividing the region into n_rows x n_cols equal cells
    and picking, per column, the row with the highest dark-fill ratio.
    More robust than contour clustering for small grids with known geometry.
    Returns a list of length n_cols: the marked row index per column, or -1.
    """
    x0, y0, rw, rh = region
    roi = binary[y0:y0 + rh, x0:x0 + rw]
    inv = cv2.bitwise_not(roi)
    cell_h = rh / n_rows
    cell_w = rw / n_cols

    result = []
    for c in range(n_cols):
        fills = []
        for r in range(n_rows):
            r0, r1 = int(r * cell_h), int((r + 1) * cell_h)
            c0, c1 = int(c * cell_w), int((c + 1) * cell_w)
            py = max(1, (r1 - r0) // 6)
            px = max(1, (c1 - c0) // 6)
            cell = inv[r0 + py:r1 - py, c0 + px:c1 - px]
            fills.append(np.sum(cell > 0) / max(cell.size, 1))
        fills = np.array(fills)
        idx = int(np.argmax(fills))
        # Marked only if clearly above the column's typical (empty) fill
        if fills[idx] > 0.04 and fills[idx] > np.median(fills) * 1.5:
            result.append(idx)
        else:
            result.append(-1)
    return result


def extract_group(page_gray):
    """
    Extract the group code (e.g. '78H') from its bubble grid.
    The grid has 2 digit columns and a separate letter column (A-J), with an
    unequal gap between them, so each part is read from its own region using a
    fixed-grid fill reader (robust to the handwritten header boxes above).
    """
    binary = preprocess(page_gray)

    xd, yd, wd, hd = _locate_grid(page_gray, GROUP_DIGITS_REGION)
    drows = _read_fixed_grid(binary, (xd, yd, wd, hd), GROUP_ROWS, 2)
    digits = "".join(str(r) if r >= 0 else "?" for r in drows)

    xl, yl, wl, hl = _locate_grid(page_gray, GROUP_LETTER_REGION)
    lrows = _read_fixed_grid(binary, (xl, yl, wl, hl), GROUP_ROWS, 1)
    letter = chr(ord('A') + lrows[0]) if lrows[0] >= 0 else "?"

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
