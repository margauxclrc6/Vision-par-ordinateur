"""
Signature matching against the class database.

Uses normalized cross-correlation (NCC) as the similarity metric.
The database directory STUDENT_CLASS_SIGNATURES must contain one image per
student named <studentID>.<ext> (jpg, png, …).

Steps:
  1. Normalize query and reference signatures to a fixed canonical size.
  2. Compute NCC between the query and every reference.
  3. Return the ID of the best match if its score exceeds the threshold.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from utils.image_processing import normalize_signature, image_similarity


# ── Tuneable parameters ────────────────────────────────────────────────────────
SIG_TARGET_SIZE = (128, 64)    # (width, height) of normalized signature
MATCH_THRESHOLD = 0.45         # minimum NCC to accept a match
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _load_database(signatures_dir):
    """
    Load and normalize all reference signatures.

    Returns
    -------
    db : dict  {student_id: normalized_image}
    """
    db = {}
    for f in Path(signatures_dir).iterdir():
        if f.suffix.lower() not in SUPPORTED_EXT:
            continue
        student_id = f.stem
        img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        db[student_id] = normalize_signature(img, SIG_TARGET_SIZE)
    return db


def match_signature(sig_gray, signatures_dir):
    """
    Identify a signature image against the class database.

    Parameters
    ----------
    sig_gray : np.ndarray
        Grayscale crop of the signature to identify.
    signatures_dir : str | Path
        Path to the directory containing one reference image per student.

    Returns
    -------
    best_id : str or None
        The matched student ID, or None if no match exceeds the threshold.
    best_score : float
        The NCC score of the best match.
    """
    db = _load_database(signatures_dir)
    if not db:
        return None, 0.0

    query = normalize_signature(sig_gray, SIG_TARGET_SIZE)

    best_id = None
    best_score = -1.0

    for student_id, ref in db.items():
        score = image_similarity(query, ref)
        if score > best_score:
            best_score = score
            best_id = student_id

    if best_score < MATCH_THRESHOLD:
        return None, best_score

    return best_id, best_score
