"""Lecture automatique des formulaires d'examen — Programme 2."""

import os
import re
from pathlib import Path

import cv2
import numpy as np
import openpyxl

from utils.pdf_utils import pdf_to_images
from utils.image_processing import deskew, preprocess
from utils.grid_reader import (extract_student_id, extract_group,
                                extract_signature_region)
from utils.signature_matcher import match_signature
from utils.cryptogram import validate_cryptograms, extract_cryptogram
from utils.ocr_reader import read_printed_field, read_printed_date
from utils.checkbox_reader import is_checked
from utils.form_layout import PAGE1_FIELDS, crop_field
from utils.exam_page_parser import parse_exam_page


# Choices available per question — up to 8 (A-H) as per spec
CHOICE_LABELS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']


def _read_field_ocr(page_gray, rel_coords, mode="printed"):
    """Crop a field and OCR it with CLAHE + adaptive threshold for robust reading."""
    crop = crop_field(page_gray, rel_coords)
    if crop.size == 0:
        return ""

    # Upscale to at least 60px height for reliable OCR
    h, w = crop.shape
    scale = max(1, 60 // max(h, 1))
    if scale > 1:
        crop = cv2.resize(crop, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    try:
        import pytesseract
    except ImportError:
        return ""

    # CLAHE to handle gray-shaded boxes, then adaptive threshold
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
    crop_enh = clahe.apply(crop)
    block = max(11, (min(crop_enh.shape) // 3) | 1)
    binary = cv2.adaptiveThreshold(crop_enh, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, block, 3)

    if mode == "date":
        cfg = "--psm 7 -c tessedit_char_whitelist=0123456789/"
        t = pytesseract.image_to_string(binary, config=cfg).strip()
        if not t:
            _, bin_otsu = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            t = pytesseract.image_to_string(bin_otsu, config=cfg).strip()
        return t

    t = pytesseract.image_to_string(binary, config="--psm 7").strip()
    if not t:
        _, bin_otsu = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        t = pytesseract.image_to_string(bin_otsu, config="--psm 7").strip()
    return t


def _checkbox_val(page_gray, rel_coords):
    """Check whether the checkbox at rel_coords is marked."""
    cell = crop_field(page_gray, rel_coords)
    if cell.size == 0:
        return 0
    checked, ratio = is_checked(cell)
    return 1 if checked else 0


def _parse_page1(page_gray, signatures_dir):
    """Extract all PAGE-01 fields and return them as a dict."""
    page_gray, _ = deskew(page_gray)

    data = {}

    data["Module"]    = _read_field_ocr(page_gray, PAGE1_FIELDS["module"])
    data["Professor"] = _read_field_ocr(page_gray, PAGE1_FIELDS["professor"])
    data["Date"]      = _read_field_ocr(page_gray, PAGE1_FIELDS["date"], mode="date")
    data["Code"]      = _read_field_ocr(page_gray, PAGE1_FIELDS["code"])

    data["Notes de cours"]      = _checkbox_val(page_gray, PAGE1_FIELDS["notes_cours"])
    data["Notes manuscrites"]   = _checkbox_val(page_gray, PAGE1_FIELDS["notes_manuscrites"])
    data["Ordinateur portable"] = _checkbox_val(page_gray, PAGE1_FIELDS["ordinateur"])
    data["Calculatrice"]        = _checkbox_val(page_gray, PAGE1_FIELDS["calculatrice"])
    data["Feuilles brouillon"]  = _checkbox_val(page_gray, PAGE1_FIELDS["feuilles_brouillon"])

    data["Note maximale"]     = _read_field_ocr(page_gray, PAGE1_FIELDS["note_maximale"])
    data["Note pour valider"] = _read_field_ocr(page_gray, PAGE1_FIELDS["note_valider"])

    sig_gray = extract_signature_region(page_gray)
    student_id_sig, sig_score = match_signature(sig_gray, signatures_dir)
    data["Validation signature"] = 1 if student_id_sig else 0
    data["_signature_id"]    = student_id_sig or ""
    data["_signature_score"] = sig_score

    data["Group"]      = extract_group(page_gray)
    data["STUDENT ID"] = extract_student_id(page_gray)

    return data


def _parse_exam_pages(pages_gray, debug_dir=None):
    """
    Parse exam answer pages (pages 3-7, index 2 onward, skipping page 2 which is blank).
    Returns a list of question-row dicts with sequential question numbers.
    """
    exam_rows = []
    q_offset = 0

    # Spec: "pages d'examens (p5→fin)" → index 4 onward (pages 1-4 = identity + preamble)
    for page_idx, page in enumerate(pages_gray[4:], start=4):
        page_gray, _ = deskew(page)
        rows = parse_exam_page(page_gray, CHOICE_LABELS,
                               page_idx=page_idx, debug_dir=debug_dir)
        for row in rows:
            row["QUESTION"] = q_offset + row["QUESTION"]
            exam_rows.append(row)
        q_offset += len(rows)

    return exam_rows


def _build_xlsx(page1_data, exam_rows, crypto_valid, xlsx_path):
    """Write the output XLSX with PAGE-01 and EXAM sheets."""
    wb = openpyxl.Workbook()

    ws1 = wb.active
    ws1.title = "PAGE-01"
    ws1.append(["Field", "Value"])
    for label, key in [
        ("Module",               "Module"),
        ("Professor",            "Professor"),
        ("Date",                 "Date"),
        ("Code",                 "Code"),
        ("Notes de cours",       "Notes de cours"),
        ("Notes manuscrites",    "Notes manuscrites"),
        ("Ordinateur portable",  "Ordinateur portable"),
        ("Calculatrice",         "Calculatrice"),
        ("Feuilles brouillon",   "Feuilles brouillon"),
        ("Note maximale",        "Note maximale"),
        ("Note pour valider",    "Note pour valider"),
        ("",                     ""),
        ("Prénom",               "Prénom"),
        ("Nom",                  "Nom"),
        ("Validation signature", "Validation signature"),
        ("Group",                "Group"),
        ("STUDENT ID",           "STUDENT ID"),
        ("Validation cryptogramme", "Validation cryptogramme"),
    ]:
        ws1.append([label, page1_data.get(key, "")])

    ws2 = wb.create_sheet(title="EXAM")
    if exam_rows:
        # Build headers: QUESTION, CHOIX A, CHOIX B, …, CHOIX H, MANTISSE, EXPOSANT, UNITE
        choice_headers = [f"CHOIX {c}" for c in CHOICE_LABELS]
        headers = ["QUESTION"] + choice_headers + ["MANTISSE", "EXPOSANT", "UNITE"]
        ws2.append(headers)
        for row in exam_rows:
            chosen = row.get("CHOIX", "")
            values = [row.get("QUESTION", "")]
            for c in CHOICE_LABELS:
                values.append(1 if chosen == c else 0)
            values.append(row.get("MANTISSE", ""))
            values.append(row.get("EXPOSANT", ""))
            values.append(row.get("UNITE", ""))
            ws2.append(values)

    wb.save(str(xlsx_path))


def autoReadFormID(pdf_path, signatures_dir, results_dir, debug=False):
    """Read one exam PDF and write the corresponding XLSX."""
    pdf_path = Path(pdf_path)
    signatures_dir = Path(signatures_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = results_dir / (pdf_path.stem + ".xlsx")
    debug_dir = str(results_dir / "debug") if debug else None

    print(f"  Processing {pdf_path.name} …")

    pages = pdf_to_images(pdf_path)
    if not pages:
        print(f"  [ERROR] No pages found in {pdf_path.name}")
        return

    page1_data = _parse_page1(pages[0], signatures_dir)

    crypto_valid, crypto_scores = validate_cryptograms(pages)
    page1_data["Validation cryptogramme"] = 1 if crypto_valid else 0
    print(f"    cryptogram valid={crypto_valid}  scores={[f'{s:.2f}' for s in crypto_scores]}")

    exam_rows = _parse_exam_pages(pages, debug_dir=debug_dir)

    _build_xlsx(page1_data, exam_rows, crypto_valid, xlsx_path)

    print(f"    → {xlsx_path.name}  "
          f"(studentID={page1_data.get('STUDENT ID', '')}  "
          f"sig={page1_data.get('_signature_id', '')}  "
          f"questions={len(exam_rows)})")

    return str(xlsx_path)


def autoReadForm(exam_pdf_dir, signatures_dir, results_dir, debug=False):
    """Process all PDFs in exam_pdf_dir."""
    exam_pdf_dir = Path(exam_pdf_dir)
    signatures_dir = Path(signatures_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(exam_pdf_dir.glob("*.pdf"))
    if not pdfs:
        print(f"[WARNING] No PDF files found in {exam_pdf_dir}")
        return

    print(f"[Programme 2] Found {len(pdfs)} PDF(s) in {exam_pdf_dir}")
    for pdf in pdfs:
        try:
            autoReadFormID(pdf, signatures_dir, results_dir, debug=debug)
        except Exception as exc:
            import traceback
            print(f"  [ERROR] {pdf.name}: {exc}")
            traceback.print_exc()

    print(f"\n[Programme 2] All results saved in {results_dir}")
