"""
Checkbox/bubble detection using morphological operations.
Binarizes the cell, closes small gaps, then checks the dark pixel ratio in the inner area.
"""

import cv2
import numpy as np
from utils.image_processing import preprocess, morpho_close


# ratio of dark pixels above which a checkbox is considered checked
FILL_RATIO_THRESHOLD = 0.15

# margin (pixels) to ignore around the border of a checkbox cell
BORDER_MARGIN = 3


def is_checked(cell_gray):
    """Check whether a checkbox cell is marked. Returns (checked, fill_ratio)."""
    if cell_gray.size == 0:
        return False, 0.0

    binary = preprocess(cell_gray)
    closed = morpho_close(binary, ksize=3)

    # ignore border artifacts
    m = BORDER_MARGIN
    inner = closed[m:-m, m:-m] if min(closed.shape) > 2 * m else closed
    inv = cv2.bitwise_not(inner)

    ratio = float(np.sum(inv > 0)) / inv.size
    return ratio > FILL_RATIO_THRESHOLD, ratio


def read_checkbox_row(page_gray, row_rel_coords, n_choices):
    """Read a row of n_choices checkboxes. Returns list of bool."""
    h, w = page_gray.shape
    x = int(row_rel_coords[0] * w)
    y = int(row_rel_coords[1] * h)
    rw = int(row_rel_coords[2] * w)
    rh = int(row_rel_coords[3] * h)

    roi = page_gray[y:y+rh, x:x+rw]
    cell_w = rw // n_choices

    results = []
    for i in range(n_choices):
        cell = roi[:, i*cell_w:(i+1)*cell_w]
        checked, _ = is_checked(cell)
        results.append(checked)
    return results
