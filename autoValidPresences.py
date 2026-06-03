"""
PROGRAMME 1 – Validation des présences

Entry point:
    autoValidPresences(exam_presences_dir, signatures_dir, results_dir)

For each photo in exam_presences_dir:
  - Read the student ID from the bubble grid          → studentID_grid
  - Extract the signature sub-image
  - Match the signature against the class database    → studentID_signature
  - Append a row to EXAM_FORMXX_PRESENCES.xlsx
"""

import os
import cv2
from pathlib import Path

import openpyxl

from utils.image_processing import load_image, deskew
from utils.grid_reader import extract_student_id, extract_signature_region
from utils.signature_matcher import match_signature


SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


# ──────────────────────────────────────────────────────────────────────────────

def autoValidID(image_path, signatures_dir, xlsx_path, results_dir):
    """
    Process one presence photo and append a row to the XLSX file.

    Parameters
    ----------
    image_path     : str | Path  — path to the presence photo.
    signatures_dir : str | Path  — class signature database directory.
    xlsx_path      : str | Path  — path to the (already opened) XLSX file.
    results_dir    : str | Path  — output directory (for debug images if needed).

    Returns
    -------
    (image_name, student_id_grid, student_id_signature)
    """
    image_path = Path(image_path)
    img_color, img_gray = load_image(image_path)

    # ── Deskew ────────────────────────────────────────────────────────────────
    img_gray, _ = deskew(img_gray)

    # ── Read student ID from bubble grid ──────────────────────────────────────
    student_id_grid = extract_student_id(img_gray)

    # ── Extract signature zone ────────────────────────────────────────────────
    sig_gray = extract_signature_region(img_gray)

    # ── Match signature against database ──────────────────────────────────────
    student_id_sig, score = match_signature(sig_gray, signatures_dir)

    print(f"  [{image_path.name}]  grid={student_id_grid}  "
          f"sig={student_id_sig}  (score={score:.3f})")

    return image_path.name, student_id_grid, student_id_sig


def autoValidPresences(exam_presences_dir, signatures_dir, results_dir):
    """
    Process all presence photos and write EXAM_FORMXX_PRESENCES.xlsx.

    Parameters
    ----------
    exam_presences_dir : str | Path
    signatures_dir     : str | Path
    results_dir        : str | Path  — created if it does not exist.
    """
    exam_presences_dir = Path(exam_presences_dir)
    signatures_dir = Path(signatures_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Determine exam name from directory name (strip _PRESENCES suffix)
    exam_name = exam_presences_dir.name.replace("_PRESENCES", "")
    xlsx_path = results_dir / f"{exam_name}_PRESENCES.xlsx"

    # Create workbook
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "PRESENCES"
    ws.append(["imageName", "studentID_grid", "studentID_signature"])

    # Collect images
    images = sorted([
        f for f in exam_presences_dir.iterdir()
        if f.suffix.lower() in SUPPORTED_EXT
    ])

    if not images:
        print(f"[WARNING] No images found in {exam_presences_dir}")

    for img_path in images:
        try:
            name, id_grid, id_sig = autoValidID(
                img_path, signatures_dir, xlsx_path, results_dir
            )
            ws.append([name, id_grid, id_sig if id_sig else ""])
        except Exception as exc:
            print(f"  [ERROR] {img_path.name}: {exc}")
            ws.append([img_path.name, "", ""])

    wb.save(str(xlsx_path))
    print(f"\n[Programme 1] Presences saved → {xlsx_path}")
    return str(xlsx_path)
