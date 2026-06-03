"""
Checkbox / bubble detection using low-level morphological operations.

Algorithm
---------
1. Extract the region of interest containing the checkbox.
2. Binarize (Otsu).
3. Apply morphological closing to merge nearby marks.
4. Compute the ratio of dark pixels in the inner part of the cell.
5. Compare against an adaptive threshold.

No high-level rectangle or circle detector is used.
"""

import cv2
import numpy as np
from utils.image_processing import preprocess, morpho_close


# Ratio of dark pixels above which a checkbox is considered checked
FILL_RATIO_THRESHOLD = 0.15

# Margin (pixels) to ignore around the border of a checkbox cell
BORDER_MARGIN = 3


def is_checked(cell_gray):
    """
    Determine whether a checkbox cell is marked.

    Parameters
    ----------
    cell_gray : np.ndarray  — grayscale crop of the checkbox cell.

    Returns
    -------
    checked : bool
    fill_ratio : float
    """
    if cell_gray.size == 0:
        return False, 0.0

    binary = preprocess(cell_gray)
    closed = morpho_close(binary, ksize=3)

    # Remove border artifacts
    m = BORDER_MARGIN
    inner = closed[m:-m, m:-m] if min(closed.shape) > 2 * m else closed
    inv = cv2.bitwise_not(inner)

    ratio = float(np.sum(inv > 0)) / inv.size
    return ratio > FILL_RATIO_THRESHOLD, ratio


def read_checkbox_row(page_gray, row_rel_coords, n_choices):
    """
    Read a row of checkboxes from an exam page.

    Parameters
    ----------
    page_gray      : full-page grayscale image.
    row_rel_coords : (x, y, w, h) as fractions of page dims — the entire
                     choice row region.
    n_choices      : number of choices in the row.

    Returns
    -------
    checked_list : list of bool, length n_choices.
    """
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
