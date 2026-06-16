"""
Matches a signature against the class database.
Uses a weighted combination of HOG, Hu Moments, and projection profiles —
more robust to geometric distortion than NCC (which fails on camera photos).
"""

from pathlib import Path
import cv2
import numpy as np
from utils.image_processing import load_image, normalize_signature


SIG_TARGET_SIZE = (128, 64)   # (width, height)
# 1:N identification accept threshold. Set near the validated 1:1 verification
# threshold (0.64): in identification the score is a max over all students, so
# the impostor competition is stronger and a permissive value (e.g. 0.30) would
# accept almost anything. 0.60 keeps only confident matches.
MATCH_THRESHOLD = 0.70        # combined score threshold

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
                 ".heic", ".heif", ".webp"}


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
        ph = np.sum(img > 0, axis=1).astype(np.float32)
        pv = np.sum(img > 0, axis=0).astype(np.float32)
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
    Identify a signature. Returns (best_id, best_score).
    best_id is None if score < MATCH_THRESHOLD.
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


# Default operating threshold, optimised on the validation set via the strict
# train/validation/test protocol in optimize_threshold.py (balanced accuracy of
# genuine-accept vs impostor-reject). Train & validation optima both gave 0.64.
VERIFY_THRESHOLD = 0.55


def signature_score(sig_gray, student_id, signatures_dir):
    """
    Raw similarity score between a query signature and the references of
    student_id (max over that student's reference signatures).
    Returns 0.0 if the student is unknown or the query is empty.
    Used by the evaluation harness to sweep the decision threshold.
    """
    key = str(signatures_dir)
    if key not in _db_cache:
        _db_cache[key] = _load_database(signatures_dir)
    db = _db_cache[key]

    refs = db.get(str(student_id), [])
    if not refs:
        return 0.0

    query = normalize_signature(sig_gray, SIG_TARGET_SIZE)
    if np.sum(query) == 0:
        return 0.0

    return max(_compare_signatures(query, ref) for ref in refs)


def verify_signature(sig_gray, student_id, signatures_dir, threshold=None):
    """
    Verify that sig_gray belongs to student_id (1:1 authentication).
    The decision threshold defaults to VERIFY_THRESHOLD but can be overridden
    (e.g. with a value optimised on a validation set).
    Returns (matched: bool, score: float).
    """
    if threshold is None:
        threshold = VERIFY_THRESHOLD
    score = signature_score(sig_gray, student_id, signatures_dir)
    return score >= threshold, score
