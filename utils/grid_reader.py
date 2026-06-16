"""
Extracts student ID and group from the bubble grid on page 1.
Each column encodes one digit/letter (bubbles 0-9 top to bottom).

Coordinate system: fractions of image width/height.
The presence photos are camera shots of a printed form, so coordinates
are approximate; the grid detector uses robust contour-based methods.
"""

import cv2
import numpy as np
from utils.image_processing import preprocess, morpho_open


# ── Region definitions (fraction of image width/height) ─────────────────────
# These cover the A4 form as seen in a roughly-centered camera photo.

STUDENT_ID_REGION = (0.73, 0.18, 0.24, 0.38)   # 5-digit ID: 5 cols × 10 rows
STUDENT_ID_DIGITS = 5
STUDENT_ID_ROWS   = 10   # rows 0-9

# Group grid: precisely located via projection. Header excluded.
# Digit col0 center x≈0.562, col1 x≈0.590, letter x≈0.647.
# Rows 0-9 span y≈0.222 (top of row0) to y≈0.458 (bottom of row9).
GROUP_DIGITS_REGION = (0.548, 0.221, 0.056, 0.237)  # 2 digit columns
GROUP_LETTER_REGION = (0.632, 0.221, 0.030, 0.237)  # 1 letter column (A-J)
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
    Uses the same low-level fixed-grid fill reader as the group grid:
    morphological removal of the printed box borders, then fill measurement
    per cell (no high-level rectangle/checkbox detection, cf. consignes §4.1).
    """
    binary = preprocess(page_gray)
    x, y, w, h = _locate_grid(page_gray, STUDENT_ID_REGION)
    cols = _read_fixed_grid(binary, (x, y, w, h), STUDENT_ID_ROWS, STUDENT_ID_DIGITS)
    return "".join(str(c) if c >= 0 else "?" for c in cols)


def _read_fixed_grid(binary, region, n_rows, n_cols):
    """
    Read a bubble grid by dividing the region into n_rows x n_cols equal cells
    and picking, per column, the row whose box contains an X mark.
    The empty box borders are removed by morphology first, so only the
    hand-drawn cross strokes contribute to the fill measurement.
    Returns a list of length n_cols: the marked row index per column, or -1.
    """
    x0, y0, rw, rh = region
    roi = binary[y0:y0 + rh, x0:x0 + rw]
    inv = cv2.bitwise_not(roi)

    cell_h = rh / n_rows
    cell_w = rw / n_cols

    # Remove straight box borders (long horizontal / vertical runs), keep X strokes
    h_len = max(5, int(cell_w * 0.55))
    v_len = max(5, int(cell_h * 0.55))
    h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (h_len, 1))
    v_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_len))
    lines = cv2.add(cv2.morphologyEx(inv, cv2.MORPH_OPEN, h_kern),
                    cv2.morphologyEx(inv, cv2.MORPH_OPEN, v_kern))
    marks = cv2.subtract(inv, lines)

    result = []
    for c in range(n_cols):
        fills = []
        for r in range(n_rows):
            r0, r1 = int(r * cell_h), int((r + 1) * cell_h)
            c0, c1 = int(c * cell_w), int((c + 1) * cell_w)
            cell = marks[r0:r1, c0:c1]
            fills.append(np.sum(cell > 0) / max(cell.size, 1))
        fills = np.array(fills)
        idx = int(np.argmax(fills))
        if fills[idx] > 0.02 and fills[idx] > np.median(fills) * 2.0:
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
    Return the signature sub-image from its fixed relative region.

    No high-level rectangle detection (cf. consignes §4.1): we crop the fixed
    SIGNATURE_REGION, then tighten the crop around the ink using horizontal and
    vertical projection profiles of the binarised stroke pixels — a low-level
    operation (thresholding + projections). Falls back to the whole region when
    too little ink is present.
    """
    x, y, w, h = _locate_grid(page_gray, SIGNATURE_REGION)
    roi = page_gray[y:y + h, x:x + w]

    _, binary = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    col_ink = np.sum(binary > 0, axis=0)
    row_ink = np.sum(binary > 0, axis=1)
    if col_ink.sum() == 0 or row_ink.sum() == 0:
        return roi

    # Keep columns/rows whose ink exceeds a small fraction of the peak
    col_thr = max(1, int(0.10 * col_ink.max()))
    row_thr = max(1, int(0.10 * row_ink.max()))
    cols = np.where(col_ink > col_thr)[0]
    rows = np.where(row_ink > row_thr)[0]
    if cols.size == 0 or rows.size == 0:
        return roi

    pad = 4
    x0 = max(0, cols[0] - pad)
    x1 = min(roi.shape[1], cols[-1] + pad)
    y0 = max(0, rows[0] - pad)
    y1 = min(roi.shape[0], rows[-1] + pad)
    interior = roi[y0:y1, x0:x1]
    return interior if interior.size > 0 else roi
