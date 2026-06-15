"""
Extracts student ID and group from the bubble grid on page 1.

Coordinates are derived from the reference canvas (2483×3510 px) used
in the original form design, converted to relative fractions so they
stay resolution-independent.

Group: 3 columns (tens digit, units digit, letter A-J), non-uniform spacing.
Student ID: 5 columns × 10 rows (digits 0-9).
"""

import cv2
import numpy as np
from utils.image_processing import preprocess


# ── Reference canvas size (from form design) ────────────────────────────────
_W = 2483
_H = 3510

# ── Per-bubble ROI layout (absolute coords → converted to relative below) ───
# Group grid: tens digit col, units digit col, letter col
# Each column: 10 bubbles spaced 92px apart starting at y=694
_GRP_Y0      = 694
_GRP_ROW_H   = 92
_GRP_CELL_W  = 52
_GRP_CELL_H  = 52
_GRP_TENS_X  = 1388
_GRP_UNITS_X = 1464
_GRP_LET_X   = 1618

# Student ID: 5 columns, each 10 bubbles
_ID_Y0     = 698
_ID_ROW_H  = 92
_ID_CELL_W = 49
_ID_CELL_H = 49
_ID_COL_XS = [1960 + col * 78 for col in range(5)]

GROUP_LETTERS = list("ABCDEFGHIJ")

SIGNATURE_REGION = (0.03, 0.23, 0.31, 0.27)   # relative (x, y, w, h)
# ─────────────────────────────────────────────────────────────────────────────


def _fill_ratio(binary, x, y, w, h, ph, pw):
    """Dark-pixel fill ratio in an absolute-coordinate bubble cell."""
    xp = int(x / _W * pw)
    yp = int(y / _H * ph)
    wp = max(1, int(w / _W * pw))
    hp = max(1, int(h / _H * ph))
    cell = binary[yp:yp + hp, xp:xp + wp]
    if cell.size == 0:
        return 0.0
    return float(np.sum(cell == 0)) / cell.size


def _read_column(binary, col_x, y0, row_h, cell_w, cell_h, ph, pw):
    """Return the row index (0-9) with the highest fill, or -1 if all empty."""
    scores = [
        _fill_ratio(binary, col_x, y0 + i * row_h, cell_w, cell_h, ph, pw)
        for i in range(10)
    ]
    best = int(np.argmax(scores))
    return best if scores[best] > 0.05 else -1


def extract_student_id(page_gray):
    """
    Extract the 5-digit numeric student ID from the bubble grid.
    Returns e.g. '63807'. Unread columns become '?'.
    """
    binary = preprocess(page_gray)
    ph, pw = page_gray.shape
    digits = []
    for col_x in _ID_COL_XS:
        d = _read_column(binary, col_x, _ID_Y0, _ID_ROW_H,
                         _ID_CELL_W, _ID_CELL_H, ph, pw)
        digits.append(str(d) if d >= 0 else "?")
    return "".join(digits)


def extract_group(page_gray):
    """
    Extract the group code (e.g. '78H') from its bubble grid.
    Format: <tens_digit><units_digit><letter A-J>
    """
    binary = preprocess(page_gray)
    ph, pw = page_gray.shape

    tens   = _read_column(binary, _GRP_TENS_X,  _GRP_Y0, _GRP_ROW_H,
                          _GRP_CELL_W, _GRP_CELL_H, ph, pw)
    units  = _read_column(binary, _GRP_UNITS_X, _GRP_Y0, _GRP_ROW_H,
                          _GRP_CELL_W, _GRP_CELL_H, ph, pw)
    letter = _read_column(binary, _GRP_LET_X,   _GRP_Y0, _GRP_ROW_H,
                          _GRP_CELL_W, _GRP_CELL_H, ph, pw)

    t = str(tens)   if tens   >= 0 else "?"
    u = str(units)  if units  >= 0 else "?"
    l = GROUP_LETTERS[letter] if 0 <= letter < len(GROUP_LETTERS) else "?"
    return t + u + l


def extract_signature_region(page_gray):
    """Crop and return the signature sub-image from page 1."""
    ph, pw = page_gray.shape
    xr, yr, wr, hr = SIGNATURE_REGION
    x = int(xr * pw); y = int(yr * ph)
    w = int(wr * pw); h = int(hr * ph)
    return page_gray[y:y + h, x:x + w]
