"""
Calibration utility — run this to generate visual debug images.
Shows exactly which regions the code is reading for each field.

Usage (Google Colab):
    !python calibrate.py /content/EXAM_FORM1_PDF/EXAM_FORM1_19283.pdf \
                         /content/debug_output
Then look at the saved PNG images in /content/debug_output/.
"""

import sys
import os
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.pdf_utils import pdf_to_images
from utils.image_processing import deskew, preprocess
from utils.form_layout import PAGE1_FIELDS
from utils.exam_page_parser import _find_bubbles, _bubbles_to_grid, FILL_RATIO_FLOOR


COLORS = {
    "module":            (255, 80,  80),
    "professor":         (80,  200, 80),
    "date":              (80,  80,  255),
    "code":              (255, 200, 0),
    "notes_cours":       (200, 0,   200),
    "notes_manuscrites": (0,   200, 200),
    "ordinateur":        (255, 128, 0),
    "calculatrice":      (128, 255, 0),
    "feuilles_brouillon":(0,   128, 255),
    "note_maximale":     (200, 200, 0),
    "note_valider":      (0,   200, 200),
    "prenom":            (255, 0,   128),
    "nom":               (128, 0,   255),
    "signature":         (0,   255, 128),
    "group":             (200, 100, 0),
    "student_id":        (0,   100, 200),
    "cryptogram":        (100, 100, 100),
}


def annotate_page1(page_gray, out_path):
    """Draw all PAGE1_FIELDS on the image and save."""
    vis = cv2.cvtColor(page_gray, cv2.COLOR_GRAY2BGR)
    h, w = page_gray.shape

    for key, (xr, yr, wr, hr) in PAGE1_FIELDS.items():
        x, y = int(xr * w), int(yr * h)
        bw, bh = int(wr * w), int(hr * h)
        color = COLORS.get(key, (200, 200, 200))
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(vis, key, (x, max(y - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

    cv2.imwrite(str(out_path), vis)
    print(f"  Page 1 annotation → {out_path}")


def annotate_exam_page(page_gray, page_idx, out_path):
    """Draw detected bubbles on an exam page."""
    vis = cv2.cvtColor(page_gray, cv2.COLOR_GRAY2BGR)
    ph, pw = page_gray.shape

    body = page_gray[int(0.15 * ph):, :]
    body_offset_y = int(0.15 * ph)
    bubbles_body = _find_bubbles(body)
    bubbles = [(cx, cy + body_offset_y, fill, bw, bh)
               for cx, cy, fill, bw, bh in bubbles_body]

    grid, row_centers, col_centers = _bubbles_to_grid(bubbles, ph, pw)

    for cx, cy, fill, bw, bh in bubbles:
        color = (0, 0, 255) if fill >= FILL_RATIO_FLOOR else (0, 180, 0)
        cv2.rectangle(vis,
                      (cx - bw // 2, cy - bh // 2),
                      (cx + bw // 2, cy + bh // 2),
                      color, 2)
        cv2.putText(vis, f"{fill:.2f}",
                    (cx - bw // 2, cy - bh // 2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

    # Draw row center lines
    for ry in row_centers:
        cv2.line(vis, (0, ry), (pw, ry), (0, 200, 255), 1)
    # Draw col center lines
    for cx in col_centers:
        cv2.line(vis, (cx, 0), (cx, ph), (255, 200, 0), 1)

    # Draw mantisse/exposant/unité search bands
    mant_x = int(0.60 * pw); mant_w = int(0.14 * pw)
    exp_x  = int(0.76 * pw); exp_w  = int(0.09 * pw)
    unit_x = int(0.87 * pw); unit_w = int(0.11 * pw)
    box_h  = max(30, int(ph * 0.045))
    for ry in row_centers:
        by = max(0, ry - box_h // 2)
        cv2.rectangle(vis, (mant_x, by), (mant_x + mant_w, by + box_h), (255, 0, 255), 2)
        cv2.rectangle(vis, (exp_x,  by), (exp_x + exp_w,   by + box_h), (0, 255, 255), 2)
        cv2.rectangle(vis, (unit_x, by), (unit_x + unit_w, by + box_h), (255, 128, 0), 2)

    cv2.putText(vis, f"Page {page_idx}  {len(bubbles)} bubbles  {len(row_centers)} rows  {len(col_centers)} cols",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imwrite(str(out_path), vis)
    print(f"  Page {page_idx} annotation → {out_path}  "
          f"({len(bubbles)} bubbles, {len(row_centers)} rows, {len(col_centers)} cols)")


def calibrate(pdf_path, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {pdf_path} …")
    pages = pdf_to_images(pdf_path)
    print(f"  {len(pages)} pages found")

    from utils.image_processing import deskew
    p0, _ = deskew(pages[0])
    annotate_page1(p0, out_dir / "page_01_fields.png")

    for i, page in enumerate(pages[1:], start=1):
        pg, _ = deskew(page)
        annotate_exam_page(pg, i + 1, out_dir / f"page_{i+1:02d}_bubbles.png")

    print(f"\nDone. Open images in: {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python calibrate.py <pdf_path> <output_dir>")
        sys.exit(1)
    calibrate(sys.argv[1], sys.argv[2])
