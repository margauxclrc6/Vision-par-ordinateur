"""
PROGRAMME 2 – Lecture automatique des formulaires

Entry point:
    autoReadForm(exam_pdf_dir, signatures_dir, results_dir)

For each PDF in exam_pdf_dir:
    autoReadFormID(pdf_path, signatures_dir, results_dir)
        → EXAM_FORMXX_abcd.xlsx  with two sheets:
             PAGE-01  (identification and administrative data)
             EXAM     (answers per question)
"""

import re
from pathlib import Path

import cv2
import numpy as np
import openpyxl

from utils.pdf_utils import pdf_to_images
from utils.image_processing import deskew
from utils.grid_reader import (extract_student_id, extract_group,
                                extract_signature_region)
from utils.signature_matcher import match_signature
from utils.cryptogram import validate_cryptograms, extract_cryptogram
from utils.ocr_reader import (read_printed_field, read_printed_date,
                               read_handwritten_text, read_handwritten_number)
from utils.checkbox_reader import is_checked
from utils.form_layout import (PAGE1_FIELDS, crop_field,
                                EXAM_CHOICES_START_X, EXAM_CHOICE_COL_W,
                                EXAM_ROW_START_Y, EXAM_ROW_H,
                                EXAM_MANTISSE_COL, EXAM_EXPOSANT_COL,
                                EXAM_UNITE_COL, N_CHOICES, CHOICE_LABELS)


# ──────────────────────────────────────────────────────────────────────────────
#  PAGE 1 parsing
# ──────────────────────────────────────────────────────────────────────────────

def _parse_page1(page_gray, signatures_dir):
    """
    Extract all PAGE-01 fields from the first page of the PDF.

    Returns a dict matching the row labels of the PAGE-01 sheet.
    """
    page_gray, _ = deskew(page_gray)
    h, w = page_gray.shape

    def crop(key):
        return crop_field(page_gray, PAGE1_FIELDS[key])

    def checkbox_val(key):
        cell = crop(key)
        checked, _ = is_checked(cell)
        return 1 if checked else 0

    data = {}

    # ── Printed info (rows 1-4) ────────────────────────────────────────────
    data["Module"]    = read_printed_field(crop("module"))
    data["Professor"] = read_printed_field(crop("professor"))
    data["Date"]      = read_printed_date(crop("date"))
    data["Code"]      = read_printed_field(crop("code"))

    # ── Checkboxes (rows 5-9) ──────────────────────────────────────────────
    data["Notes de cours"]      = checkbox_val("notes_cours")
    data["Notes manuscrites"]   = checkbox_val("notes_manuscrites")
    data["Ordinateur portable"] = checkbox_val("ordinateur")
    data["Calculatrice"]        = checkbox_val("calculatrice")
    data["Feuilles brouillon"]  = checkbox_val("feuilles_brouillon")

    # ── Printed scores (rows 10-11) ────────────────────────────────────────
    data["Note maximale"]    = read_printed_field(crop("note_maximale"))
    data["Note pour valider"] = read_printed_field(crop("note_valider"))

    # ── Handwritten name (rows 13-14) ──────────────────────────────────────
    data["Prénom"] = read_handwritten_text(crop("prenom"))
    data["Nom"]    = read_handwritten_text(crop("nom"))

    # ── Signature (row 15) ─────────────────────────────────────────────────
    sig_gray = crop("signature")
    student_id_sig, sig_score = match_signature(sig_gray, signatures_dir)
    data["Validation signature"] = 1 if student_id_sig else 0
    data["_signature_id"] = student_id_sig or ""
    data["_signature_score"] = sig_score

    # ── Bubble grids (rows 16-17) ──────────────────────────────────────────
    data["Group"]      = extract_group(page_gray)
    data["STUDENT ID"] = extract_student_id(page_gray)

    return data


# ──────────────────────────────────────────────────────────────────────────────
#  Exam pages parsing
# ──────────────────────────────────────────────────────────────────────────────

def _detect_questions(page_gray):
    """
    Detect question rows on an exam page using horizontal line detection.

    Returns a list of (question_number, y_center_ratio) tuples.
    """
    h, w = page_gray.shape
    rows = []

    # Simple approach: scan rows in the answer area for horizontal separators
    y_start = int(EXAM_ROW_START_Y * h)
    row_h_px = int(EXAM_ROW_H * h)

    # Estimate number of questions by available vertical space
    n_rows = int((h - y_start) / row_h_px)

    for i in range(n_rows):
        y_center = (y_start + i * row_h_px + row_h_px // 2) / h
        rows.append((i + 1, y_center))

    return rows


def _parse_exam_pages(pages_gray):
    """
    Parse exam answer pages (pages 5 onward, 0-indexed page 4+).

    Returns a list of dicts, one per question row.
    """
    exam_rows = []

    for page in pages_gray[4:]:  # pages 5 → end  (0-indexed: 4→end)
        page_gray, _ = deskew(page)
        h, w = page_gray.shape

        questions = _detect_questions(page_gray)

        for q_num, y_ratio in questions:
            row = {"QUESTION": q_num}

            # ── Multiple-choice checkboxes ─────────────────────────────────
            for ci, label in enumerate(CHOICE_LABELS):
                x_ratio = EXAM_CHOICES_START_X + ci * EXAM_CHOICE_COL_W
                x = int(x_ratio * w)
                y = int((y_ratio - EXAM_ROW_H / 2) * h)
                cell_w = int(EXAM_CHOICE_COL_W * w)
                cell_h = int(EXAM_ROW_H * h)
                cell = page_gray[max(0, y):y+cell_h, x:x+cell_w]
                checked, _ = is_checked(cell)
                row[f"CHOIX {label}"] = 1 if checked else ""

            # ── Handwritten numerical answer ───────────────────────────────
            def _read_col(rel_col):
                cx = int(rel_col[0] * w)
                cy = int((y_ratio - EXAM_ROW_H / 2) * h)
                cw = int(rel_col[2] * w)
                ch = int(rel_col[3] * h / len(questions) if questions else EXAM_ROW_H * h)
                cell = page_gray[max(0, cy):cy+cell_h, cx:cx+cw]
                return read_handwritten_number(cell)

            row["MANTISSE"] = _read_col(EXAM_MANTISSE_COL)
            row["EXPOSANT"] = _read_col(EXAM_EXPOSANT_COL)
            row["UNITE"]    = _read_col(EXAM_UNITE_COL)

            exam_rows.append(row)

    return exam_rows


# ──────────────────────────────────────────────────────────────────────────────
#  Excel builder
# ──────────────────────────────────────────────────────────────────────────────

def _build_xlsx(page1_data, exam_rows, crypto_valid, xlsx_path):
    """Create the output XLSX with PAGE-01 and EXAM sheets."""
    wb = openpyxl.Workbook()

    # ── PAGE-01 sheet ──────────────────────────────────────────────────────
    ws1 = wb.active
    ws1.title = "PAGE-01"

    page1_rows = [
        ("Module",              page1_data.get("Module", "")),
        ("Professor",           page1_data.get("Professor", "")),
        ("Date",                page1_data.get("Date", "")),
        ("Code",                page1_data.get("Code", "")),
        ("Notes de cours",      page1_data.get("Notes de cours", "")),
        ("Notes manuscrites",   page1_data.get("Notes manuscrites", "")),
        ("Ordinateur portable", page1_data.get("Ordinateur portable", "")),
        ("Calculatrice",        page1_data.get("Calculatrice", "")),
        ("Feuilles brouillon",  page1_data.get("Feuilles brouillon", "")),
        ("Note maximale",       page1_data.get("Note maximale", "")),
        ("Note pour valider",   page1_data.get("Note pour valider", "")),
        ("",                    ""),
        ("Prénom",              page1_data.get("Prénom", "")),
        ("Nom",                 page1_data.get("Nom", "")),
        ("Validation signature",page1_data.get("Validation signature", "")),
        ("Group",               page1_data.get("Group", "")),
        ("STUDENT ID",          page1_data.get("STUDENT ID", "")),
        ("Validation cryptogramme", 1 if crypto_valid else 0),
    ]
    ws1.append(["Field", "Value"])
    for label, value in page1_rows:
        ws1.append([label, value])

    # ── EXAM sheet ─────────────────────────────────────────────────────────
    ws2 = wb.create_sheet(title="EXAM")

    if exam_rows:
        headers = list(exam_rows[0].keys())
        ws2.append(headers)
        for row in exam_rows:
            ws2.append([row.get(h, "") for h in headers])

    wb.save(str(xlsx_path))


# ──────────────────────────────────────────────────────────────────────────────
#  Public API
# ──────────────────────────────────────────────────────────────────────────────

def autoReadFormID(pdf_path, signatures_dir, results_dir):
    """
    Read one exam PDF and write the corresponding XLSX.

    Parameters
    ----------
    pdf_path       : str | Path
    signatures_dir : str | Path
    results_dir    : str | Path
    """
    pdf_path = Path(pdf_path)
    signatures_dir = Path(signatures_dir)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    xlsx_path = results_dir / (pdf_path.stem + ".xlsx")

    print(f"  Processing {pdf_path.name} …")

    # ── Convert PDF to images ──────────────────────────────────────────────
    pages = pdf_to_images(pdf_path)

    if not pages:
        print(f"  [ERROR] No pages found in {pdf_path.name}")
        return

    # ── Parse page 1 ──────────────────────────────────────────────────────
    page1_data = _parse_page1(pages[0], signatures_dir)

    # ── Validate cryptograms ───────────────────────────────────────────────
    crypto_valid, crypto_scores = validate_cryptograms(pages)
    page1_data["Validation cryptogramme"] = 1 if crypto_valid else 0
    print(f"    cryptogram valid={crypto_valid}  scores={[f'{s:.2f}' for s in crypto_scores]}")

    # ── Parse exam answer pages ────────────────────────────────────────────
    exam_rows = _parse_exam_pages(pages)

    # ── Write XLSX ─────────────────────────────────────────────────────────
    _build_xlsx(page1_data, exam_rows, crypto_valid, xlsx_path)

    print(f"    → {xlsx_path.name}  "
          f"(studentID={page1_data.get('STUDENT ID','')}  "
          f"sig={page1_data.get('_signature_id','')})")

    return str(xlsx_path)


def autoReadForm(exam_pdf_dir, signatures_dir, results_dir):
    """
    Process all PDFs in exam_pdf_dir.

    Parameters
    ----------
    exam_pdf_dir   : str | Path
    signatures_dir : str | Path
    results_dir    : str | Path
    """
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
            autoReadFormID(pdf, signatures_dir, results_dir)
        except Exception as exc:
            print(f"  [ERROR] {pdf.name}: {exc}")

    print(f"\n[Programme 2] All results saved in {results_dir}")
