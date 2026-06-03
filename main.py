"""
MAIN – DeepForm exam auto-correction pipeline

Configure the two variables below and run:
    python main.py

The script will:
  1. Run Programme 1 (autoValidPresences) → EXAM_FORMXX_PRESENCES.xlsx
  2. Run Programme 2 (autoReadForm)       → one XLSX per scanned PDF
"""

import os
from pathlib import Path
from autoValidPresences import autoValidPresences
from autoReadForm import autoReadForm


# ── INPUT CONFIGURATION ────────────────────────────────────────────────────────
# Only these two lines need to change for the challenge dataset.

EXAM_NAME = "EXAM_FORM01"
SIGNATURES_DIR = "STUDENT_CLASS_SIGNATURES"

# ── DERIVED PATHS (do not modify) ─────────────────────────────────────────────
PRESENCES_DIR = f"{EXAM_NAME}_PRESENCES"
PDF_DIR       = f"{EXAM_NAME}_PDF"
RESULTS_DIR   = f"{EXAM_NAME}_RESULTS"


def main():
    # Create results directory
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  DeepForm — Exam: {EXAM_NAME}")
    print("=" * 60)

    # ── Programme 1 ─────────────────────────────────────────────────────
    if Path(PRESENCES_DIR).exists():
        print(f"\n[Programme 1] Validating presences from: {PRESENCES_DIR}")
        autoValidPresences(PRESENCES_DIR, SIGNATURES_DIR, RESULTS_DIR)
    else:
        print(f"\n[Programme 1] SKIPPED — directory not found: {PRESENCES_DIR}")

    # ── Programme 2 ─────────────────────────────────────────────────────
    if Path(PDF_DIR).exists():
        print(f"\n[Programme 2] Reading forms from: {PDF_DIR}")
        autoReadForm(PDF_DIR, SIGNATURES_DIR, RESULTS_DIR)
    else:
        print(f"\n[Programme 2] SKIPPED — directory not found: {PDF_DIR}")

    print("\n[Done] Results are in:", RESULTS_DIR)


if __name__ == "__main__":
    main()
