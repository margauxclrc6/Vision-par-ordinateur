"""Validation des présences — lit le formulaire, extrait l'ID et la signature."""

import os
import cv2
from pathlib import Path

import openpyxl

from utils.image_processing import load_image, deskew
from utils.grid_reader import extract_student_id, extract_signature_region
from utils.signature_matcher import match_signature


SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def autoValidID(image_path, signatures_dir, xlsx_path, results_dir):
    """Process one presence photo and return (image_name, id_grid, id_sig)."""
    image_path = Path(image_path)
    img_color, img_gray = load_image(image_path)

    img_gray, _ = deskew(img_gray)
    student_id_grid = extract_student_id(img_gray)
    sig_gray = extract_signature_region(img_gray)
    student_id_sig, score = match_signature(sig_gray, signatures_dir)

    print(f"  [{image_path.name}]  grid={student_id_grid}  "
          f"sig={student_id_sig}  (score={score:.3f})")

    return image_path.name, student_id_grid, student_id_sig


def autoValidPresences(exam_presences_dir, signatures_dir, results_dir):
    """Process all presence photos and write EXAM_FORMXX_PRESENCES.xlsx."""
    exam_presences_dir = Path(exam_presences_dir)
    signatures_dir = Path(signatures_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    exam_name = exam_presences_dir.name.replace("_PRESENCES", "")
    xlsx_path = results_dir / f"{exam_name}_PRESENCES.xlsx"

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "PRESENCES"
    ws.append(["imageName", "studentID_grid", "studentID_signature"])

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
