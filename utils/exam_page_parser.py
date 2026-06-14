"""
Exam page parser — block-segmentation approach.

The real form layout (discovered from debug images):
  • Each question sits inside its own full-width bordered rectangle.
  • Multiple-choice questions have their checkboxes stacked VERTICALLY in a
    single column on the left margin (x ≈ 0.08-0.13). Choice A is on top,
    then B, C, D, (E). The marked box is filled with an X / cross.
  • Numerical questions have NO left checkbox column; instead they have
    handwritten boxes in the centre-left: a mantissa box, a "×10" label,
    an exponent box, and a unit box.

Strategy:
  1. Detect question blocks via full-width horizontal border lines (morphology).
  2. For each block:
       - look for checkbox bubbles in the left strip
       - if found  → multiple-choice: the filled bubble's vertical index = choice
       - if absent → numerical: OCR the mantissa / exponent / unit boxes
  3. Return one row per question, in top-to-bottom (page) order.
"""

import cv2
import numpy as np
from utils.image_processing import (preprocess, morpho_open,
                                     find_horizontal_lines)


# ── tunables ──────────────────────────────────────────────────────────────────
MIN_BUBBLE_AREA_RATIO = 0.00015  # min checkbox area as fraction of page area
MAX_BUBBLE_AREA_RATIO = 0.004    # max checkbox area as fraction of page area
ASPECT_RATIO_RANGE    = (0.3, 3.0)   # width/height of a checkbox bounding box

# Left strip where multiple-choice checkboxes live (fraction of page width)
CHECKBOX_X_MIN = 0.04
CHECKBOX_X_MAX = 0.16

# Skip top fraction of each block (contains "● QUESTION N" header band)
BLOCK_HEADER_SKIP = 0.25

# Fill detection
FILL_FLOOR     = 0.14    # absolute dark-pixel ratio above which a box is "marked"
FILL_RELATIVE  = 1.6     # marked box must be ≥ this × the median fill of its group

# Question-block detection
BORDER_MIN_LEN_RATIO = 0.35   # a border line must span ≥ 35 % of the page width
BLOCK_MIN_HEIGHT_RATIO = 0.03 # a question block must be ≥ 3 % of the page height
HEADER_SKIP_RATIO = 0.08      # ignore the top 8 % (page header band)
FOOTER_SKIP_RATIO = 0.04      # ignore the bottom 4 % (page number / cryptogram)

# Numerical answer-box positions (fraction of page width)
# One student-fill rectangle (value) at far left; unit label pre-printed on right
MANT_X, MANT_W = 0.03, 0.13   # the student-written value box
EXP_X,  EXP_W  = 0.16, 0.08   # small exponent box (above mantissa row)
UNIT_X, UNIT_W = 0.30, 0.22   # pre-printed unit label box
# ─────────────────────────────────────────────────────────────────────────────


def _detect_question_blocks(page_gray):
    """
    Return a list of (y0, y1) tuples, one per question block, top-to-bottom.
    Blocks are the tall regions enclosed by full-width horizontal border lines.
    """
    ph, pw = page_gray.shape
    binary = preprocess(page_gray)

    h_mask = find_horizontal_lines(binary, min_len_ratio=BORDER_MIN_LEN_RATIO)
    row_strength = np.sum(h_mask > 0, axis=1)          # how "line-like" each row is
    line_thresh = BORDER_MIN_LEN_RATIO * pw
    is_line = row_strength >= line_thresh

    # Cluster consecutive line rows into single border y-positions
    borders = []
    y = 0
    while y < ph:
        if is_line[y]:
            y_start = y
            while y < ph and is_line[y]:
                y += 1
            borders.append((y_start + y - 1) // 2)
        else:
            y += 1

    # Regions between consecutive borders that are tall enough = question blocks
    min_h = BLOCK_MIN_HEIGHT_RATIO * ph
    y_top_limit = HEADER_SKIP_RATIO * ph
    y_bot_limit = (1 - FOOTER_SKIP_RATIO) * ph

    blocks = []
    for i in range(len(borders) - 1):
        y0, y1 = borders[i], borders[i + 1]
        if (y1 - y0) >= min_h and y0 >= y_top_limit and y1 <= y_bot_limit:
            blocks.append((y0, y1))

    return blocks


def _find_checkboxes_in_strip(page_gray, y0, y1):
    """
    Find checkbox bubbles inside the left strip of the [y0, y1] block.
    Returns list of (cy, fill_ratio) sorted top-to-bottom.
    """
    ph, pw = page_gray.shape
    page_area = ph * pw

    x0 = int(CHECKBOX_X_MIN * pw)
    x1 = int(CHECKBOX_X_MAX * pw)
    # Skip the header band at the top of the block (contains the ● QUESTION N label)
    y_skip = int(y0 + (y1 - y0) * BLOCK_HEADER_SKIP)
    strip = page_gray[y_skip:y1, x0:x1]
    if strip.size == 0:
        return []

    binary = preprocess(strip)
    inv = cv2.bitwise_not(binary)
    opened = morpho_open(inv, ksize=2)

    cnts, _ = cv2.findContours(opened, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    min_area = MIN_BUBBLE_AREA_RATIO * page_area
    max_area = MAX_BUBBLE_AREA_RATIO * page_area

    best_per_slot = {}   # key → (cy, fill)
    for cnt in cnts:
        bx, by, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if not (min_area < area < max_area):
            continue
        aspect = bw / max(bh, 1)
        if not (ASPECT_RATIO_RANGE[0] < aspect < ASPECT_RATIO_RANGE[1]):
            continue
        pad = max(2, int(min(bw, bh) * 0.18))
        inner = inv[by + pad: by + bh - pad, bx + pad: bx + bw - pad]
        fill = float(np.sum(inner > 0)) / max(inner.size, 1)
        cy = y_skip + by + bh // 2
        key = (by // 20,)
        if key not in best_per_slot or fill > best_per_slot[key][1]:
            best_per_slot[key] = (cy, fill)

    boxes = sorted(best_per_slot.values(), key=lambda b: b[0])
    return boxes


def _read_number_box(page_gray, x, y, w, h, letters=False):
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
    _, binary_check = cv2.threshold(crop, 0, 255,
                                    cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dark_ratio = np.sum(binary_check > 0) / max(binary_check.size, 1)
    if dark_ratio < 0.005:
        print(f"    DBG _read_number_box: SKIP empty (dark={dark_ratio:.4f}) x={x} y={y} w={w} h={h}")
        return ""
    scale = max(1, 80 // max(crop.shape[0], 1))
    crop_up = cv2.resize(crop, (crop.shape[1] * scale, crop.shape[0] * scale),
                         interpolation=cv2.INTER_CUBIC)
    _, binary = cv2.threshold(crop_up, 0, 255,
                              cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    whitelist = ("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
                 if letters else "0123456789.-")
    text = pytesseract.image_to_string(
        binary, config=f"--psm 8 -c tessedit_char_whitelist={whitelist}")
    return text.strip()


def parse_exam_page(page_gray, choice_labels=None, page_idx=0, debug_dir=None):
    """
    Parse one exam page using block segmentation.
    Returns list of dicts: [{QUESTION, CHOIX, MANTISSE, EXPOSANT, UNITE}, ...]
    """
    if choice_labels is None:
        choice_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    ph, pw = page_gray.shape
    blocks = _detect_question_blocks(page_gray)

    debug_items = []   # (y0, y1, checkboxes, marked_idx, kind)
    rows = []

    for (y0, y1) in blocks:
        checkboxes = _find_checkboxes_in_strip(page_gray, y0, y1)

        row = {"QUESTION": 0, "CHOIX": "", "MANTISSE": "", "EXPOSANT": "", "UNITE": ""}

        if len(checkboxes) >= 3:
            # ── Multiple-choice question ──
            fills = np.array([f for (_, f) in checkboxes])
            best = int(np.argmax(fills))
            print(f"  DBG Q{len(rows)+1}: {len(checkboxes)} boxes, fills={[round(f,3) for _,f in checkboxes]}, best={best}")
            med = float(np.median(fills))
            marked = None
            if fills[best] >= FILL_FLOOR and fills[best] >= med * FILL_RELATIVE:
                marked = best
                if best < len(choice_labels):
                    row["CHOIX"] = choice_labels[best]
                else:
                    row["CHOIX"] = str(best)
            debug_items.append((y0, y1, checkboxes, marked, "mcq"))
        else:
            # ── Numerical question ──
            box_h  = max(30, int((y1 - y0) * 0.28))
            by_num = max(0, y1 - box_h - 5)
            m = _read_number_box(page_gray, int(MANT_X * pw), by_num,
                                 int(MANT_W * pw), box_h)
            print(f"  DBG NUM Q{len(rows)+1}: block=[{y0},{y1}] by_num={by_num} box_h={box_h} mant={m!r}")
            row["MANTISSE"] = m
            row["EXPOSANT"] = _read_number_box(page_gray, int(EXP_X * pw), by_num,
                                               int(EXP_W * pw), box_h)
            row["UNITE"]    = _read_number_box(page_gray, int(UNIT_X * pw), by_num,
                                               int(UNIT_W * pw), box_h, letters=True)
            debug_items.append((y0, y1, checkboxes, None, "num"))

        rows.append(row)

    # Number questions 1..N in page order
    for i, row in enumerate(rows):
        row["QUESTION"] = i + 1

    if debug_dir is not None:
        _save_debug(page_gray, debug_items, debug_dir, page_idx)

    return rows


def _save_debug(page_gray, debug_items, debug_dir, page_idx):
    """Save an annotated image showing detected blocks, checkboxes and marks."""
    import os
    os.makedirs(debug_dir, exist_ok=True)
    ph, pw = page_gray.shape
    vis = cv2.cvtColor(page_gray, cv2.COLOR_GRAY2BGR)

    x0 = int(CHECKBOX_X_MIN * pw)
    x1 = int(CHECKBOX_X_MAX * pw)

    for (y0, y1, checkboxes, marked, kind) in debug_items:
        # block outline
        bcol = (255, 150, 0) if kind == "mcq" else (200, 0, 200)
        cv2.rectangle(vis, (2, y0), (pw - 3, y1), bcol, 1)
        # checkboxes
        for idx, (cy, fill) in enumerate(checkboxes):
            col = (0, 0, 255) if idx == marked else (0, 200, 0)
            cv2.rectangle(vis, (x0, cy - 10), (x1, cy + 10), col, 2)
            cv2.putText(vis, f"{fill:.2f}", (x1 + 4, cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)
        if kind == "num":
            cv2.putText(vis, "NUM", (x0, y0 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 0, 200), 2)

    out = os.path.join(debug_dir, f"page_{page_idx:02d}_blocks.png")
    cv2.imwrite(out, vis)
