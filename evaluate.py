"""
Évaluation quantitative des programmes autoValidPresences et autoReadForm.
Compare les sorties XLSX avec les vérités terrain.

Usage (Colab):
    !python evaluate.py /content/EXAM_FORM1_RESULTS /content/drive/.../GT_FORM1

Structure attendue :
    results_dir/
        EXAM_FORM1_PRESENCES.xlsx      ← sortie Programme 1
        EXAM_FORM1_63807.xlsx          ← sortie Programme 2 (un par PDF)
    ground_truth_dir/
        EXAM_FORM1_PRESENCES.xlsx      ← vérité terrain Programme 1
        EXAM_FORM1_63807.xlsx          ← vérité terrain Programme 2
"""

import sys
from pathlib import Path
import openpyxl


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_sheet(xlsx_path, sheet=0):
    wb = openpyxl.load_workbook(str(xlsx_path), read_only=True, data_only=True)
    ws = wb.worksheets[sheet] if isinstance(sheet, int) else wb[sheet]
    rows = list(ws.iter_rows(values_only=True))
    if not rows:
        return []
    headers = [str(h).strip() if h is not None else "" for h in rows[0]]
    return [{headers[i]: row[i] for i in range(len(headers))} for row in rows[1:]]


def _norm(v):
    """Normalize cell value: strip, upper, remove spaces, cast numeric."""
    s = str(v).strip().upper().replace(" ", "") if v is not None else ""
    try:
        return str(int(float(s)))
    except Exception:
        return s


# ── Programme 1 — présences ──────────────────────────────────────────────────

def evaluate_presences(pred_xlsx, gt_xlsx):
    pred_rows = _load_sheet(pred_xlsx, 0)
    gt_rows   = _load_sheet(gt_xlsx,   0)
    if not pred_rows or not gt_rows:
        print("  [WARN] Feuille présences vide — ignorée")
        return {}

    gt_by_name = {str(r.get("imageName", r.get("IMAGE", ""))).strip(): r
                  for r in gt_rows}

    total = correct_grid = correct_sig = 0
    errors = []

    for row in pred_rows:
        name = str(row.get("imageName", "")).strip()
        gt = gt_by_name.get(name)
        if gt is None:
            continue
        total += 1
        p_grid = _norm(row.get("studentID_grid", ""))
        p_sig  = _norm(row.get("studentID_signature", ""))
        g_grid = _norm(gt.get("studentID_grid", gt.get("STUDENT_ID", "")))
        g_sig  = _norm(gt.get("studentID_signature", gt.get("STUDENT_ID_SIG", "")))

        ok_grid = p_grid == g_grid
        ok_sig  = p_sig  == g_sig
        if ok_grid: correct_grid += 1
        if ok_sig:  correct_sig  += 1
        if not ok_grid or not ok_sig:
            errors.append(f"      {name}: grille={p_grid}(GT={g_grid})  "
                          f"sig={p_sig}(GT={g_sig})")

    acc_g = correct_grid / total if total else 0
    acc_s = correct_sig  / total if total else 0
    print(f"  Présences — {total} images")
    print(f"    ID grille    : {correct_grid}/{total}  ({acc_g*100:.1f}%)")
    print(f"    ID signature : {correct_sig}/{total}  ({acc_s*100:.1f}%)")
    for e in errors:
        print(e)
    return {"n": total, "grid_acc": acc_g, "sig_acc": acc_s}


# ── Programme 2 — formulaires d'examen ───────────────────────────────────────

CHOICE_LABELS = list("ABCDEFGH")

PAGE1_FIELDS_TO_CHECK = [
    "Module", "Professor", "Date", "Code",
    "Notes de cours", "Notes manuscrites", "Ordinateur portable",
    "Calculatrice", "Feuilles brouillon",
    "Note maximale", "Note pour valider",
    "Prénom", "Nom", "Group", "STUDENT ID",
    "Validation signature", "Validation cryptogramme",
]


def evaluate_exam(pred_xlsx, gt_xlsx):
    print(f"  {Path(pred_xlsx).name}")

    # PAGE-01
    try:
        pred_p1 = {str(r.get("Field", "")).strip(): str(r.get("Value", "")).strip()
                   for r in _load_sheet(pred_xlsx, 0)}
        gt_p1   = {str(r.get("Field", "")).strip(): str(r.get("Value", "")).strip()
                   for r in _load_sheet(gt_xlsx,   0)}
    except Exception as e:
        print(f"    [WARN] PAGE-01 erreur: {e}")
        pred_p1, gt_p1 = {}, {}

    p1_total = p1_correct = 0
    p1_errors = []
    for field in PAGE1_FIELDS_TO_CHECK:
        if field not in gt_p1:
            continue
        p1_total += 1
        if _norm(pred_p1.get(field, "")) == _norm(gt_p1[field]):
            p1_correct += 1
        else:
            p1_errors.append(f"      {field}: prédit={pred_p1.get(field, '')!r}  "
                             f"GT={gt_p1[field]!r}")

    acc_p1 = p1_correct / p1_total if p1_total else 0
    print(f"    PAGE-01 : {p1_correct}/{p1_total} champs corrects ({acc_p1*100:.1f}%)")
    for e in p1_errors:
        print(e)

    # EXAM sheet
    try:
        pred_exam = _load_sheet(pred_xlsx, 1)
        gt_exam   = _load_sheet(gt_xlsx,   1)
    except Exception as e:
        print(f"    [WARN] Feuille EXAM erreur: {e}")
        return {"p1_acc": acc_p1, "choice_acc": 0.0, "q_detect_rate": 0.0}

    gt_by_q   = {_norm(r.get("QUESTION", "")): r for r in gt_exam   if r.get("QUESTION")}
    pred_by_q = {_norm(r.get("QUESTION", "")): r for r in pred_exam if r.get("QUESTION")}

    n_gt   = len(gt_by_q)
    n_pred = len(pred_by_q)
    q_rate = min(n_pred, n_gt) / n_gt if n_gt else 0

    choice_total = choice_correct = 0
    choice_errors = []

    for q, gt_row in gt_by_q.items():
        pred_row = pred_by_q.get(q)

        gt_choice   = next((c for c in CHOICE_LABELS
                            if _norm(gt_row.get(f"CHOIX {c}", "0")) == "1"), None)
        pred_choice = next((c for c in CHOICE_LABELS
                            if pred_row and _norm(pred_row.get(f"CHOIX {c}", "0")) == "1"),
                           None) if pred_row else None

        choice_total += 1
        if gt_choice == pred_choice:
            choice_correct += 1
        else:
            tag = "(non détectée)" if pred_row is None else ""
            choice_errors.append(f"      Q{q}: prédit={pred_choice!r}  GT={gt_choice!r} {tag}")

    acc_c = choice_correct / choice_total if choice_total else 0
    print(f"    EXAM : {n_pred} questions détectées (GT={n_gt}, taux={q_rate*100:.1f}%)")
    print(f"    CHOIX : {choice_correct}/{choice_total} corrects ({acc_c*100:.1f}%)")
    for e in choice_errors:
        print(e)

    return {"p1_acc": acc_p1, "q_detect_rate": q_rate, "choice_acc": acc_c}


# ── main ─────────────────────────────────────────────────────────────────────

def evaluate(results_dir, ground_truth_dir):
    results_dir      = Path(results_dir)
    ground_truth_dir = Path(ground_truth_dir)

    print("=" * 60)
    print("  ÉVALUATION QUANTITATIVE — DeepForm")
    print("=" * 60)

    all_grid = []
    all_sig  = []
    all_p1   = []
    all_q    = []
    all_c    = []

    # Programme 1
    pres_files = sorted(results_dir.glob("*PRESENCES*.xlsx"))
    if pres_files:
        print("\n── Programme 1 : Présences ──────────────────────────")
        for pred in pres_files:
            gt = ground_truth_dir / pred.name
            if not gt.exists():
                print(f"  [SKIP] GT introuvable : {pred.name}")
                continue
            m = evaluate_presences(pred, gt)
            if m:
                all_grid.append(m["grid_acc"])
                all_sig.append(m["sig_acc"])

    # Programme 2
    exam_files = sorted(f for f in results_dir.glob("*.xlsx")
                        if "PRESENCES" not in f.name)
    if exam_files:
        print("\n── Programme 2 : Formulaires d'examen ───────────────")
        for pred in exam_files:
            gt = ground_truth_dir / pred.name
            if not gt.exists():
                print(f"  [SKIP] GT introuvable : {pred.name}")
                continue
            m = evaluate_exam(pred, gt)
            all_p1.append(m.get("p1_acc", 0))
            all_q.append(m.get("q_detect_rate", 0))
            all_c.append(m.get("choice_acc", 0))
            print()

    # Synthèse
    def avg(lst):
        return f"{sum(lst)/len(lst)*100:.1f}%" if lst else "N/A"

    print("=" * 60)
    print("  SYNTHÈSE")
    print("=" * 60)
    if all_grid: print(f"  ID grille (prog.1)      : {avg(all_grid)}")
    if all_sig:  print(f"  ID signature (prog.1)   : {avg(all_sig)}")
    if all_p1:   print(f"  Champs PAGE-01 (prog.2) : {avg(all_p1)}")
    if all_q:    print(f"  Taux détection questions : {avg(all_q)}")
    if all_c:    print(f"  Précision choix A-H     : {avg(all_c)}")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate.py <results_dir> <ground_truth_dir>")
        sys.exit(1)
    evaluate(sys.argv[1], sys.argv[2])
