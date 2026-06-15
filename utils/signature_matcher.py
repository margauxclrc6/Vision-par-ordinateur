"""
Matches a signature against the class database.
Uses a weighted combination of HOG, Hu Moments, and projection profiles
(ported from reference implementation).
"""

from pathlib import Path
import cv2
import numpy as np
from utils.image_processing import load_image, normalize_signature


SIG_TARGET_SIZE = (128, 64)   # (width, height)
MATCH_THRESHOLD = 0.45        # combined score threshold

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
                 ".heic", ".heif", ".webp"}


# ── Descriptor functions ──────────────────────────────────────────────────────

def _sig_score_hog(a, b):
    def hog_hist(img):
        gx = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(img.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        hist, _ = np.histogram(ang[mag > 10], bins=8, range=(0, 360),
                               weights=mag[mag > 10])
        norm = np.linalg.norm(hist)
        return hist / norm if norm > 0 else hist
    return float(np.dot(hog_hist(a), hog_hist(b)))


def _sig_score_moments(a, b):
    def hu(img):
        m = cv2.moments(img)
        h = cv2.HuMoments(m).flatten()
        return -np.sign(h) * np.log10(np.abs(h) + 1e-10)
    return 1.0 / (1.0 + np.sum(np.abs(hu(a) - hu(b))))


def _sig_score_projection(a, b):
    def profiles(img):
        ph = np.sum(img == 0, axis=1).astype(np.float32)
        pv = np.sum(img == 0, axis=0).astype(np.float32)
        ph /= ph.max() + 1e-6
        pv /= pv.max() + 1e-6
        return np.concatenate([ph, pv])
    corr = np.corrcoef(profiles(a), profiles(b))[0, 1]
    return float(max(0.0, corr))


def _compare_signatures(a, b):
    """Weighted combination: HOG 50% + Projection 30% + Moments 20%."""
    return (0.5 * _sig_score_hog(a, b)
            + 0.3 * _sig_score_projection(a, b)
            + 0.2 * _sig_score_moments(a, b))


# ── Database loading ──────────────────────────────────────────────────────────

def _load_database(signatures_dir):
    db = {}

    def _add(path, student_id):
        try:
            _, img = load_image(path)
        except Exception:
            return
        norm = normalize_signature(img, SIG_TARGET_SIZE)
        if np.sum(norm) > 0:
            db.setdefault(student_id, []).append(norm)

    def _walk(folder, current_id):
        for item in sorted(Path(folder).iterdir()):
            if item.is_dir():
                sid = item.name if item.name.isdigit() else current_id
                _walk(item, sid)
            elif item.suffix.lower() in SUPPORTED_EXT:
                sid = current_id or item.stem
                if sid:
                    _add(item, sid)

    _walk(signatures_dir, None)
    return db


_db_cache = {}


def match_signature(sig_gray, signatures_dir):
    """
    Identify a signature against the class database.
    Returns (best_id, best_score). best_id is None if score < MATCH_THRESHOLD.
    """
    key = str(signatures_dir)
    if key not in _db_cache:
        _db_cache[key] = _load_database(signatures_dir)
    db = _db_cache[key]

    if not db:
        return None, 0.0

    query = normalize_signature(sig_gray, SIG_TARGET_SIZE)
    if np.sum(query) == 0:
        return None, 0.0

    best_id, best_score = None, -1.0
    for student_id, refs in db.items():
        score = max(_compare_signatures(query, ref) for ref in refs)
        if score > best_score:
            best_score = score
            best_id = student_id

    if best_score < MATCH_THRESHOLD:
        return None, best_score
    return best_id, best_score
