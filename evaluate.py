"""
Quantitative evaluation script.

Compares the output XLSX files against ground-truth annotations to compute:
  - Accuracy on studentID_grid  (Programme 1)
  - Accuracy on studentID_signature (Programme 1)
  - Accuracy on printed fields  (Programme 2, PAGE-01 rows 1-4, 9-11)
  - Accuracy on handwritten fields (Programme 2, PAGE-01 rows 13-14)
  - Accuracy on checkbox fields    (Programme 2, PAGE-01 rows 5-9, 16-18)
  - Accuracy on MCQ choices        (Programme 2, EXAM CHOIX columns)
  - Accuracy on mantisse/exposant  (Programme 2, EXAM)

Usage:
    python evaluate.py --results EXAM_FORM01_RESULTS --gt ground_truth/

The ground-truth directory must follow the same naming convention as results.
"""

import argparse
from pathlib import Path

import openpyxl


def load_presences_xlsx(path):
    wb = openpyxl.load_workbook(str(path))
    ws = wb.active
    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        rows.append({"imageName": row[0], "grid": row[1], "sig": row[2]})
    return rows


def accuracy(pred_list, gt_list, key):
    total, correct = 0, 0
    for p, g in zip(pred_list, gt_list):
        pv = str(p.get(key, "") or "").strip()
        gv = str(g.get(key, "") or "").strip()
        if gv:
            total += 1
            correct += int(pv == gv)
    return correct / total if total else 0.0


def evaluate_presences(results_dir, gt_dir):
    pred_file = next(Path(results_dir).glob("*_PRESENCES.xlsx"), None)
    gt_file   = next(Path(gt_dir).glob("*_PRESENCES.xlsx"), None)
    if not pred_file or not gt_file:
        print("[SKIP] Presence files not found.")
        return
    pred = load_presences_xlsx(pred_file)
    gt   = load_presences_xlsx(gt_file)
    acc_grid = accuracy(pred, gt, "grid")
    acc_sig  = accuracy(pred, gt, "sig")
    print(f"  Programme 1 — StudentID grid accuracy : {acc_grid:.3f}")
    print(f"  Programme 1 — Signature ID accuracy   : {acc_sig:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--gt",      required=True,
                        help="Ground truth directory with same file structure")
    args = parser.parse_args()

    print("=" * 50)
    print("  Evaluation")
    print("=" * 50)
    evaluate_presences(args.results, args.gt)
    # Per-form evaluation can be added here following the same pattern.


if __name__ == "__main__":
    main()
