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

STUDENT_ID_REGION = (0.70, 0.16, 0.27, 0.30)   # generous search zone; the grid
# itself is then localised inside it via _locate_bubble_grid (checkbox detection)
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


def _locate_bubble_grid(binary, search_region, n_rows, n_cols):
    """
    Localise précisément la grille de cases à cocher à l'intérieur d'une zone de
    recherche généreuse, pour s'affranchir du cadrage variable des photos.

    Bas niveau : on détecte les contours carrés des cases (taille homogène),
    puis on renvoie le cadre englobant exact de la matrice n_rows x n_cols
    (centres extrêmes ± une demi-cellule). Si la détection n'est pas fiable
    (moins de 40 % des cases trouvées), on retombe sur la zone de recherche fixe.
    """
    x0, y0, sw, sh = search_region
    roi = binary[y0:y0 + sh, x0:x0 + sw]
    inv = cv2.bitwise_not(roi)  # cases = traits clairs sur fond sombre
    cnts, _ = cv2.findContours(inv, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    approx_cell = sw / max(n_cols, 1)
    cand = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if not (approx_cell * 0.20 < w < approx_cell * 0.90):
            continue
        if not (approx_cell * 0.20 < h < approx_cell * 0.90):
            continue
        ar = w / float(h)
        if ar < 0.6 or ar > 1.7:
            continue
        cand.append((x + w / 2.0, y + h / 2.0, w, h))

    if len(cand) < n_rows * n_cols * 0.4:
        return search_region

    med_w = np.median([c[2] for c in cand])
    med_h = np.median([c[3] for c in cand])
    cand = [c for c in cand
            if 0.6 * med_w < c[2] < 1.5 * med_w
            and 0.6 * med_h < c[3] < 1.5 * med_h]
    if len(cand) < n_rows * n_cols * 0.4:
        return search_region

    xs = np.array([c[0] for c in cand])
    ys = np.array([c[1] for c in cand])
    xmin, xmax, ymin, ymax = xs.min(), xs.max(), ys.min(), ys.max()

    cellw = (xmax - xmin) / (n_cols - 1) if n_cols > 1 else med_w
    cellh = (ymax - ymin) / (n_rows - 1) if n_rows > 1 else med_h

    gx = int(x0 + xmin - cellw / 2)
    gy = int(y0 + ymin - cellh / 2)
    gw = int((xmax - xmin) + cellw)
    gh = int((ymax - ymin) + cellh)

    if gw < sw * 0.2 or gh < sh * 0.2:
        return search_region
    return (gx, gy, gw, gh)


def extract_student_id(page_gray):
    """
    Extract the numeric student ID from the bubble grid.
    Returns e.g. '63807' or a string with '?' for unread columns.
    The grid is first localised by detecting the checkbox squares (robust to the
    variable framing of camera photos), then read with the fixed-grid fill reader.
    """
    binary = preprocess(page_gray)
    search = _locate_grid(page_gray, STUDENT_ID_REGION)
    region = _locate_bubble_grid(binary, search, STUDENT_ID_ROWS, STUDENT_ID_DIGITS)
    cols = _read_fixed_grid(binary, region, STUDENT_ID_ROWS, STUDENT_ID_DIGITS)
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
