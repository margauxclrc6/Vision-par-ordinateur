"""
Signature-threshold optimisation with a strict train / validation / test
protocol (methodology — Section 4.2).

The signature decision threshold is:
  • OPTIMISED on the training set,
  • SELECTED (cross-checked) on the validation set,
  • and the final performance is REPORTED on the held-out test set,
which is never used for tuning.

Ground truth is read automatically from file names (utils.ground_truth), so no
manual annotation is needed. For the signature axis we build:
  • genuine pairs  : (photo signature, its true ID)        → should ACCEPT
  • impostor pairs : (photo signature, another student ID) → should REJECT
and optimise the threshold for balanced accuracy — exactly the trade-off the
challenge probes by injecting identity-usurpation cases.

Usage:
    python optimize_threshold.py TRAIN_DIR VAL_DIR TEST_DIR \
           --signatures STUDENT_CLASS_SIGNATURES
e.g.
    python optimize_threshold.py /content/EXAM_FORM1_PRESENCES \
        /content/EXAM_FORM2_PRESENCES /content/EXAM_FORM3_PRESENCES \
        --signatures /content/STUDENT_CLASS_SIGNATURES
"""

import argparse
import random

import cv2
import numpy as np

from utils.image_processing import load_image, deskew, correct_perspective
from utils.grid_reader import extract_student_id, extract_signature_region
from utils.signature_matcher import signature_score
from utils.ground_truth import list_labelled_images


def _prepare(img_gray):
    """Standardise resolution and deskew, mirroring autoValidPresences."""
    h, w = img_gray.shape
    if h > 2000:
        scale = 2000 / h
        img_gray = cv2.resize(img_gray, (int(w * scale), 2000),
                              interpolation=cv2.INTER_AREA)
    img_gray, _ = deskew(img_gray)
    corrected, ok = correct_perspective(img_gray)
    if ok:
        img_gray = corrected
    return img_gray


def collect_scores(presences_dir, signatures_dir, seed=0):
    """
    Run the pipeline once over a presences directory and collect, per image:
      • grid_correct   : was the bubble-grid student ID read correctly?
      • genuine_score  : signature score against the true ID
      • impostor_score : signature score against a random different ID
    """
    rng = random.Random(seed)
    pairs = list_labelled_images(presences_dir)
    all_ids = [gt for _, gt in pairs]

    records = []
    for path, gt in pairs:
        try:
            _, gray = load_image(path)
        except Exception:
            continue
        gray = _prepare(gray)

        grid_id = extract_student_id(gray)
        sig = extract_signature_region(gray)

        genuine = signature_score(sig, gt, signatures_dir)
        others = [i for i in all_ids if i != gt]
        impostor_id = rng.choice(others) if others else gt
        impostor = signature_score(sig, impostor_id, signatures_dir)

        records.append({
            "name": path.name, "gt": gt, "grid_id": grid_id,
            "grid_correct": grid_id == gt,
            "genuine_score": genuine, "impostor_score": impostor,
        })
    return records


def grid_accuracy(records):
    if not records:
        return 0.0
    return sum(r["grid_correct"] for r in records) / len(records)


def balanced_accuracy(records, threshold):
    """0.5 * (genuine-accept rate + impostor-reject rate) at a threshold."""
    if not records:
        return 0.0
    gen = np.array([r["genuine_score"] for r in records])
    imp = np.array([r["impostor_score"] for r in records])
    tpr = np.mean(gen >= threshold)
    tnr = np.mean(imp < threshold)
    return 0.5 * (tpr + tnr)


def optimise_threshold(records, grid=None):
    if grid is None:
        grid = np.round(np.arange(0.30, 0.80, 0.01), 3)
    best_t, best_acc = grid[0], -1.0
    for t in grid:
        acc = balanced_accuracy(records, t)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return best_t, best_acc


def _report(name, records, threshold):
    gen = np.array([r["genuine_score"] for r in records])
    imp = np.array([r["impostor_score"] for r in records])
    print(f"\n── {name}  ({len(records)} images) ──")
    print(f"  StudentID grid accuracy   : {grid_accuracy(records):6.1%}")
    print(f"  Signature balanced acc.   : {balanced_accuracy(records, threshold):6.1%}"
          f"  (threshold={threshold:.2f})")
    print(f"    genuine  score mean={gen.mean():.3f} median={np.median(gen):.3f}")
    print(f"    impostor score mean={imp.mean():.3f} median={np.median(imp):.3f}")
    print(f"    genuine accept rate (TPR) : {np.mean(gen >= threshold):.1%}")
    print(f"    impostor reject rate (TNR): {np.mean(imp <  threshold):.1%}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("train"); ap.add_argument("val"); ap.add_argument("test")
    ap.add_argument("--signatures", required=True)
    args = ap.parse_args()

    print("Collecting scores (train/val/test) …")
    train = collect_scores(args.train, args.signatures)
    val   = collect_scores(args.val,   args.signatures)
    test  = collect_scores(args.test,  args.signatures)

    t_train, acc_train = optimise_threshold(train)
    print(f"\n[TRAIN] optimal threshold = {t_train:.2f} (balanced acc {acc_train:.1%})")

    t_val, acc_val = optimise_threshold(val)
    print(f"[VALIDATION] own optimum = {t_val:.2f} ({acc_val:.1%}); "
          f"train threshold scores {balanced_accuracy(val, t_train):.1%} here")

    operating_t = round((t_train + t_val) / 2, 2)
    print(f"\n>>> Selected operating threshold = {operating_t:.2f}")
    print(f"    (set utils/signature_matcher.VERIFY_THRESHOLD = {operating_t:.2f})")

    for name, recs in [("TRAIN", train), ("VALIDATION", val), ("TEST", test)]:
        _report(name, recs, operating_t)


if __name__ == "__main__":
    main()
