"""Validation des présences — lit le formulaire, extrait l'ID et la signature."""

from pathlib import Path
import cv2
import numpy as np
import openpyxl

from utils.image_processing import load_image, deskew, correct_perspective
from utils.grid_reader import extract_student_id, extract_signature_region
from utils.signature_matcher import match_signature, verify_signature


# All image extensions that may appear in the presences folder
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
                 ".heic", ".heif", ".webp", ".JPG", ".PNG", ".JPEG"}


def _prepare_image(img_gray):
    """
    Prepare a presence photo for processing:
    1. Resize to a standard height (avoids memory issues with 12 MP photos)
    2. Deskew
    """
    # Cap height at 2000 px to standardise resolution
    h, w = img_gray.shape
    if h > 2000:
        scale = 2000 / h
        img_gray = cv2.resize(img_gray, (int(w * scale), 2000),
                              interpolation=cv2.INTER_AREA)

    img_gray, _ = deskew(img_gray)

    # NB: an L-bracket perspective correction was explored (see
    # utils.image_processing.correct_perspective) but, on these fairly frontal
    # camera photos, the deskew already removes most distortion and the warp did
    # not improve grid accuracy on the validation set, so it is left disabled.
    return img_gray


def autoValidID(image_path, signatures_dir, xlsx_path, results_dir):
    """
    Process one presence photo.
    Returns (image_name, student_id_grid, student_id_signature).
    student_id_signature is None if no match found.
    """
    image_path = Path(image_path)
    _, img_gray = load_image(image_path)
    img_gray = _prepare_image(img_gray)

    student_id_grid = extract_student_id(img_gray)
    sig_gray = extract_signature_region(img_gray)

    # Column C (studentID_signature = StudentID_bitmap, cf. §3.3): the identity
    # the signature itself belongs to, deduced from the signature database —
    # INDEPENDENT of the grid, so the professor can compare B and C to spot an
    # identity usurpation (B != C) or an unrecognised signature (C empty).
    #
    # We first test the fast, accurate 1:1 hypothesis "the signature confirms the
    # grid ID"; if it is not confirmed (wrong/usurped/unreadable grid), we fall
    # back to a full 1:N identification so column C is still populated.
    verified, score = verify_signature(sig_gray, student_id_grid, signatures_dir)
    if verified and "?" not in student_id_grid:
        student_id_sig = student_id_grid
    else:
        student_id_sig, score = match_signature(sig_gray, signatures_dir)

    status = "✓" if (student_id_sig and student_id_sig == student_id_grid) else "?"
    print(f"  {status} [{image_path.name}]  "
          f"grid={student_id_grid}  sig={student_id_sig}  (score={score:.3f})")

    return image_path.name, student_id_grid, student_id_sig


def autoValidPresences(exam_presences_dir, signatures_dir, results_dir):
    """
    Process all presence photos in exam_presences_dir.
    Writes EXAM_FORMXX_PRESENCES.xlsx with columns:
        imageName | studentID_grid | studentID_signature
    """
    exam_presences_dir = Path(exam_presences_dir)
    signatures_dir     = Path(signatures_dir)
    results_dir        = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    exam_name = exam_presences_dir.name.replace("_PRESENCES", "")
    xlsx_path = results_dir / f"{exam_name}_PRESENCES.xlsx"

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "PRESENCES"
    ws.append(["imageName", "studentID_grid", "studentID_signature"])

    # Collect all image files (case-insensitive extension match)
    images = sorted([
        f for f in exam_presences_dir.iterdir()
        if f.suffix.lower() in {e.lower() for e in SUPPORTED_EXT}
    ])

    if not images:
        print(f"[WARNING] No images found in {exam_presences_dir}")

    ok = wrong = unmatched = 0
    for img_path in images:
        try:
            name, id_grid, id_sig = autoValidID(
                img_path, signatures_dir, xlsx_path, results_dir)
            ws.append([name, id_grid, id_sig if id_sig else ""])
            if id_sig and id_sig == id_grid:
                ok += 1
            elif id_sig is None:
                unmatched += 1
            else:
                wrong += 1
        except Exception as exc:
            print(f"  [ERROR] {img_path.name}: {exc}")
            ws.append([img_path.name, "", ""])

    wb.save(str(xlsx_path))
    print(f"\n[Programme 1] {len(images)} images — "
          f"{ok} OK / {unmatched} non reconnues / {wrong} suspectes")
    print(f"  → {xlsx_path}")
    return str(xlsx_path)
