"""
Smart exam page parser.
Automatically detects the answer table structure without hardcoded pixel coordinates.
Works by:
  1. Finding the answer grid (bubbles) via contour detection — restricted to left 58% of page
  2. Clustering bubbles into rows (questions) and columns (choices)
  3. Filtering out noise by requiring each valid row/column to contain ≥ MIN_MEMBERS bubbles
  4. Reading mantisse/exposant/unité boxes on the right side
"""

import cv2
import numpy as np
from collections import Counter
from utils.image_processing import preprocess, morpho_open, morpho_close


# ── tunables ──────────────────────────────────────────────────────────────────
MIN_BUBBLE_AREA_RATIO = 0.0002   # min bubble area as fraction of page area
MAX_BUBBLE_AREA_RATIO = 0.004    # max bubble area as fraction of page area
ASPECT_RATIO_RANGE    = (0.35, 2.8)   # width/height of bubble bounding box
Y_CLUSTER_GAP_RATIO   = 0.025    # min vertical gap (as page-height ratio) between rows
X_CLUSTER_GAP_RATIO   = 0.015    # min horizontal gap (as page-width ratio) between cols
FILL_RATIO_FLOOR      = 0.10     # minimum dark-pixel ratio to count as "filled"
FILL_RATIO_RELATIVE   = 1.4      # filled must be >= this × median fill in its row

# Bubble grid is in the left portion of the page (right side = mantisse/exposant/unité boxes)
BUBBLE_X_MAX_RATIO = 0.58        # ignore bubble candidates beyond this x fraction

# Grid quality filter: a row/column is valid only if it has at least this many bubbles
MIN_BUBBLES_PER_ROW = 2          # a question row must have ≥ 2 detected choices
MIN_BUBBLES_PER_COL = 2          # a choice column must appear in ≥ 2 question rows
# ─────────────────────────────────────────────────────────────────────────────


def _find_bubbles(page_gray):
    """
    Return list of (cx, cy, fill_ratio, w, h) for every bubble-like contour
    in the left portion of the page (where answer bubbles live).
    """
    ph, pw = page_gray.shape
    page_area = ph * pw

    binary = preprocess(page_gray)
    inv = cv2.bitwise_not(binary)
    opened = morpho_open(inv, ksize=2)

    cnts, _ = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    min_area = MIN_BUBBLE_AREA_RATIO * page_area
    max_area = MAX_BUBBLE_AREA_RATIO * page_area
    x_limit = int(BUBBLE_X_MAX_RATIO * pw)

    bubbles = []
    seen = set()
    for cnt in cnts:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if not (min_area < area < max_area):
            continue
        aspect = bw / max(bh, 1)
        if not (ASPECT_RATIO_RANGE[0] < aspect < ASPECT_RATIO_RANGE[1]):
            continue
        cx = bx + bw // 2
        # Only keep bubbles in the left bubble-grid area
        if cx > x_limit:
            continue
        key = (bx // 12, by // 12)
        if key in seen:
            continue
        seen.add(key)
        cy = by + bh // 2
        pad = max(2, int(min(bw, bh) * 0.12))
        inner = inv[by + pad: by + bh - pad, bx + pad: bx + bw - pad]
        fill = float(np.sum(inner > 0)) / max(inner.size, 1)
        bubbles.append((cx, cy, fill, bw, bh))

    return bubbles


def _cluster_1d(values, gap):
    """Group sorted values into clusters separated by at least `gap`."""
    if not values:
        return []
    values = sorted(values)
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] > gap:
            clusters.append([])
        clusters[-1].append(v)
    return [int(np.median(c)) for c in clusters]


def _bubbles_to_grid(bubbles, ph, pw):
    """
    Convert flat bubble list → dict {(row_idx, col_idx): fill_ratio}.
    Filters out rows/columns that don't have enough bubbles (noise reduction).
    Returns (grid_dict, row_centers_y, col_centers_x).
    """
    if not bubbles:
        return {}, [], []

    y_gap = Y_CLUSTER_GAP_RATIO * ph
    x_gap = X_CLUSTER_GAP_RATIO * pw

    cy_list = [b[1] for b in bubbles]
    cx_list = [b[0] for b in bubbles]

    row_centers_all = _cluster_1d(cy_list, y_gap)
    col_centers_all = _cluster_1d(cx_list, x_gap)

    def nearest(val, centers):
        return int(np.argmin([abs(val - c) for c in centers]))

    # First pass: assign every bubble to its nearest row/col center
    assignments = []
    for cx, cy, fill, bw, bh in bubbles:
        r = nearest(cy, row_centers_all)
        c = nearest(cx, col_centers_all)
        assignments.append((r, c, fill))

    # Count how many bubbles each row and column index received
    row_counts = Counter(a[0] for a in assignments)
    col_counts = Counter(a[1] for a in assignments)

    # Keep only rows/cols with enough members — this drops table border hits, etc.
    valid_rows = sorted(r for r, cnt in row_counts.items() if cnt >= MIN_BUBBLES_PER_ROW)
    valid_cols = sorted(c for c, cnt in col_counts.items() if cnt >= MIN_BUBBLES_PER_COL)

    if not valid_rows or not valid_cols:
        # Fallback: no filtering (rare edge case)
        valid_rows = list(range(len(row_centers_all)))
        valid_cols = list(range(len(col_centers_all)))

    # Remap row/col indices to 0-based after filtering
    row_remap = {old: new for new, old in enumerate(valid_rows)}
    col_remap = {old: new for new, old in enumerate(valid_cols)}

    row_centers = [row_centers_all[i] for i in valid_rows]
    col_centers = [col_centers_all[i] for i in valid_cols]

    grid = {}
    for r, c, fill in assignments:
        if r in row_remap and c in col_remap:
            nr, nc = row_remap[r], col_remap[c]
            if (nr, nc) not in grid or fill > grid[(nr, nc)]:
                grid[(nr, nc)] = fill

    return grid, row_centers, col_centers


def _assign_choices(grid, row_centers, col_centers, choice_labels):
    """
    Decide which choice is marked per row.
    Returns list of (choice_label or None) per row.
    """
    n_rows = len(row_centers)
    n_cols = len(col_centers)
    answers = []
    n_choice = len(choice_labels)

    if n_cols >= n_choice:
        for r in range(n_rows):
            fills = np.array([grid.get((r, c), 0.0) for c in range(n_cols)])
            choice_fills = fills[:n_choice]
            best = int(np.argmax(choice_fills))
            med = float(np.median(choice_fills))
            if choice_fills[best] >= FILL_RATIO_FLOOR and choice_fills[best] >= med * FILL_RATIO_RELATIVE:
                answers.append(choice_labels[best] if best < len(choice_labels) else str(best))
            else:
                answers.append(None)
        return answers, "cols"

    elif n_cols == 1 or (n_rows % n_choice == 0):
        n_questions = n_rows // n_choice if n_choice > 0 else n_rows
        for q in range(n_questions):
            fills = np.array([grid.get((q * n_choice + ci, 0), 0.0)
                              for ci in range(n_choice)])
            best = int(np.argmax(fills))
            med = float(np.median(fills))
            if fills[best] >= FILL_RATIO_FLOOR and fills[best] >= med * FILL_RATIO_RELATIVE:
                answers.append(choice_labels[best] if best < len(choice_labels) else str(best))
            else:
                answers.append(None)
        return answers, "rows"

    else:
        for r in range(n_rows):
            fills = np.array([grid.get((r, c), 0.0) for c in range(n_cols)])
            best = int(np.argmax(fills))
            med = float(np.median(fills))
            if fills[best] >= FILL_RATIO_FLOOR and fills[best] >= med * FILL_RATIO_RELATIVE:
                answers.append(choice_labels[best] if best < len(choice_labels) else str(best))
            else:
                answers.append(None)
        return answers, "cols_mixed"


def _read_number_box(page_gray, x, y, w, h):
    """OCR a small handwritten-number box. Returns cleaned string. Skips empty boxes."""
    try:
        import pytesseract
    except ImportError:
        return ""
    if w <= 0 or h <= 0:
        return ""
    crop = page_gray[max(0, y):y + h, max(0, x):x + w]
    if crop.size == 0:
        return ""
    # Quick emptiness check — skip OCR if box has no dark pixels
    _, binary_check = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if np.sum(binary_check > 0) / max(binary_check.size, 1) < 0.02:
        return ""
    # upscale for better OCR
    scale = max(1, 80 // max(crop.shape[0], 1))
    crop_up = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale),
                         interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(crop_up, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(
        binary,
        config="--psm 7 -c tessedit_char_whitelist=0123456789.,-"
    )
    return text.strip()


def parse_exam_page(page_gray, choice_labels=None, page_idx=0, debug_dir=None):
    """
    Parse one exam page.
    Returns list of dicts: [{QUESTION, CHOICE, MANTISSE, EXPOSANT, UNITE}, ...]
    choice_labels: e.g. ['A','B','C','D']
    """
    if choice_labels is None:
        choice_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    ph, pw = page_gray.shape

    # Skip the top ~15 % (header) and look only at the body
    body = page_gray[int(0.15 * ph):, :]
    body_offset_y = int(0.15 * ph)

    bubbles_body = _find_bubbles(body)
    # Adjust y-coordinates back to full-page reference
    bubbles = [(cx, cy + body_offset_y, fill, bw, bh)
               for cx, cy, fill, bw, bh in bubbles_body]

    if debug_dir is not None:
        _save_debug(page_gray, bubbles, debug_dir, page_idx)

    grid, row_centers, col_centers = _bubbles_to_grid(bubbles, ph, pw)

    if not row_centers:
        return []

    answers, interpretation = _assign_choices(grid, row_centers, col_centers, choice_labels)

    rows = []
    n_questions = len(answers)
    for q_idx, choice in enumerate(answers):
        row = {
            "QUESTION": q_idx + 1,
            "CHOIX": choice or "",
        }
        if q_idx < len(row_centers):
            ry = row_centers[q_idx]
        else:
            ry = int((q_idx + 0.5) / max(n_questions, 1) * ph)

        box_h = max(30, int(ph * 0.045))
        box_y = max(0, ry - box_h // 2)

        mant_x = int(0.60 * pw)
        mant_w = int(0.14 * pw)
        exp_x  = int(0.76 * pw)
        exp_w  = int(0.09 * pw)
        unit_x = int(0.87 * pw)
        unit_w = int(0.11 * pw)

        row["MANTISSE"] = _read_number_box(page_gray, mant_x, box_y, mant_w, box_h)
        row["EXPOSANT"] = _read_number_box(page_gray, exp_x,  box_y, exp_w,  box_h)
        row["UNITE"]    = _read_number_box(page_gray, unit_x, box_y, unit_w, box_h)

        rows.append(row)

    return rows


def _save_debug(page_gray, bubbles, debug_dir, page_idx):
    """Save annotated debug image showing detected bubbles."""
    import os
    os.makedirs(debug_dir, exist_ok=True)
    vis = cv2.cvtColor(page_gray, cv2.COLOR_GRAY2BGR)
    for cx, cy, fill, bw, bh in bubbles:
        color = (0, 0, 255) if fill > FILL_RATIO_FLOOR else (0, 200, 0)
        cv2.rectangle(vis, (cx - bw // 2, cy - bh // 2),
                      (cx + bw // 2, cy + bh // 2), color, 2)
        cv2.putText(vis, f"{fill:.2f}", (cx - bw // 2, cy - bh // 2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    out = os.path.join(debug_dir, f"page_{page_idx:02d}_bubbles.png")
    cv2.imwrite(out, vis)
