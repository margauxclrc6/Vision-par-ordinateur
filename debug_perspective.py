"""
Debug visuel de la correction de perspective.

Pour chaque photo : sauvegarde
  • les 4 ROIs de coin avec le L-bracket détecté (point rouge),
  • l'image redressée (warp) avec les régions calibrées superposées
    (grille ID, groupe, signature), pour vérifier qu'elles tombent au bon endroit.

Usage (Colab) :
    !python debug_perspective.py /content/EXAM_FORM1_PRESENCES --out /content/debug --n 4
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

from utils.image_processing import (
    load_image, deskew, correct_perspective, _find_corner_mark,
)
from utils.grid_reader import (
    STUDENT_ID_REGION, GROUP_DIGITS_REGION, GROUP_LETTER_REGION,
    SIGNATURE_REGION,
)


def _prepare(img_gray):
    h, w = img_gray.shape
    if h > 2000:
        scale = 2000 / h
        img_gray = cv2.resize(img_gray, (int(w * scale), 2000),
                              interpolation=cv2.INTER_AREA)
    img_gray, _ = deskew(img_gray)
    return img_gray


def _draw_corner_debug(img_gray):
    """Reproduit la logique de correct_perspective et dessine les détections."""
    h, w = img_gray.shape
    margin_x = int(w * 0.18)
    margin_y = int(h * 0.18)
    rois = {
        'tl': (img_gray[:margin_y, :margin_x],            0,            0),
        'tr': (img_gray[:margin_y, w - margin_x:],         w - margin_x, 0),
        'bl': (img_gray[h - margin_y:, :margin_x],         0,            h - margin_y),
        'br': (img_gray[h - margin_y:, w - margin_x:],     w - margin_x, h - margin_y),
    }
    vis = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    all_found = True
    for key, (roi, ox, oy) in rois.items():
        # cadre du ROI
        cv2.rectangle(vis, (ox, oy), (ox + roi.shape[1], oy + roi.shape[0]),
                      (0, 180, 0), 3)
        pt = _find_corner_mark(roi, key)
        if pt is None:
            all_found = False
            cv2.putText(vis, f"{key}: NONE", (ox + 10, oy + 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            continue
        gx, gy = pt[0] + ox, pt[1] + oy
        cv2.circle(vis, (gx, gy), 18, (0, 0, 255), -1)
        cv2.putText(vis, key, (gx + 20, gy),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
    return vis, all_found


def _draw_regions(warped):
    """Superpose les régions calibrées sur l'image redressée."""
    vis = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    h, w = warped.shape
    regions = {
        "ID":     (STUDENT_ID_REGION,    (0, 0, 255)),
        "GRP_D":  (GROUP_DIGITS_REGION,  (255, 0, 0)),
        "GRP_L":  (GROUP_LETTER_REGION,  (255, 128, 0)),
        "SIG":    (SIGNATURE_REGION,     (0, 200, 0)),
    }
    for name, (reg, col) in regions.items():
        x = int(reg[0] * w); y = int(reg[1] * h)
        bw = int(reg[2] * w); bh = int(reg[3] * h)
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), col, 2)
        cv2.putText(vis, name, (x, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("presences_dir")
    ap.add_argument("--out", default="debug_perspective")
    ap.add_argument("--n", type=int, default=4, help="nb d'images à traiter")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".bmp", ".heic", ".heif", ".webp"}
    images = sorted(f for f in Path(args.presences_dir).iterdir()
                    if f.suffix.lower() in exts)[:args.n]

    for path in images:
        try:
            _, gray = load_image(path)
        except Exception as e:
            print(f"[skip] {path.name}: {e}")
            continue
        gray = _prepare(gray)

        corner_vis, found = _draw_corner_debug(gray)
        cv2.imwrite(str(out / f"{path.stem}_1_corners.png"), corner_vis)

        warped, ok = correct_perspective(gray)
        status = "OK" if ok else "FAIL(returns input)"
        if ok:
            cv2.imwrite(str(out / f"{path.stem}_2_warped.png"),
                        _draw_regions(warped))
        print(f"{path.name:35s} corners_found={found}  warp={status}")

    print(f"\n→ images dans {out}/  (regarde *_1_corners.png et *_2_warped.png)")


if __name__ == "__main__":
    main()
