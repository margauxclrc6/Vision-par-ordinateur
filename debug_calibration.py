"""
Calibration script: renders one exam page with all reading rectangles drawn.
Run in Colab, then display/download the output image to verify coordinates.
"""

import cv2
import numpy as np
from utils.pdf_utils import pdf_to_images
from utils.image_processing import deskew
from utils.exam_page_parser import (
    _detect_question_blocks, MANT_X, MANT_W, EXP_X, EXP_W, UNIT_X, UNIT_W
)
from utils.form_layout import PAGE1_FIELDS, crop_field


def draw_rect(img, x, y, w, h, color, label=""):
    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    if label:
        cv2.putText(img, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


def calibrate_page1(pdf_path, out_path="calibration_page1.png"):
    pages = pdf_to_images(pdf_path)
    page = pages[0]
    page, _ = deskew(page)
    ph, pw = page.shape
    vis = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)

    colors = {
        "module": (255, 0, 0), "professor": (255, 0, 0),
        "date": (255, 0, 0), "code": (255, 0, 0),
        "notes_cours": (0, 200, 0), "notes_manuscrites": (0, 200, 0),
        "ordinateur": (0, 200, 0), "calculatrice": (0, 200, 0),
        "feuilles_brouillon": (0, 200, 0),
        "note_maximale": (0, 0, 255), "note_valider": (0, 0, 255),
        "group": (255, 128, 0), "student_id": (255, 128, 0),
        "signature": (128, 0, 255), "cryptogram": (0, 128, 255),
    }
    for name, rel in PAGE1_FIELDS.items():
        xr, yr, wr, hr = rel
        x, y, w, h = int(xr*pw), int(yr*ph), int(wr*pw), int(hr*ph)
        draw_rect(vis, x, y, w, h, colors.get(name, (100, 100, 100)), name)

    cv2.imwrite(out_path, vis)
    print(f"Saved: {out_path}")


def calibrate_exam_page(pdf_path, page_index=4, out_path="calibration_exam.png"):
    """page_index=4 = first exam page (0-indexed)."""
    pages = pdf_to_images(pdf_path)
    if page_index >= len(pages):
        print(f"Page {page_index} not found (PDF has {len(pages)} pages)")
        return
    page, _ = deskew(pages[page_index])
    ph, pw = page.shape
    vis = cv2.cvtColor(page, cv2.COLOR_GRAY2BGR)

    blocks = _detect_question_blocks(page)
    print(f"Page {page_index}: {len(blocks)} blocks detected")

    for i, (y0, y1) in enumerate(blocks):
        # Block outline
        cv2.rectangle(vis, (0, y0), (pw - 1, y1), (200, 200, 0), 1)
        cv2.putText(vis, f"Q{i+1}", (4, y0 + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 0), 2)

        box_h  = max(30, int((y1 - y0) * 0.25))
        by_num = max(0, y1 - box_h - 3)
        exp_h  = max(20, int((y1 - y0) * 0.14))
        by_exp = max(0, by_num - exp_h - 4)

        # Mantissa (cyan)
        draw_rect(vis, int(MANT_X*pw), by_num, int(MANT_W*pw), box_h,
                  (255, 255, 0), "MANT")
        # Exponent (orange)
        draw_rect(vis, int(EXP_X*pw), by_exp, int(EXP_W*pw), exp_h,
                  (0, 165, 255), "EXP")
        # Unite (green)
        draw_rect(vis, int(UNIT_X*pw), by_num, int(UNIT_W*pw), box_h,
                  (0, 255, 0), "UNIT")

    cv2.imwrite(out_path, vis)
    print(f"Saved: {out_path}  (page dims: {pw}x{ph})")


# ── Run ──────────────────────────────────────────────────────────────────────
PDF = "EXAM_FORM1_63807.pdf"   # ← change to your actual PDF path

calibrate_page1(PDF)
calibrate_exam_page(PDF, page_index=4)   # first exam page
# calibrate_exam_page(PDF, page_index=5) # second exam page if needed
