"""
Matches a signature against the class database using normalized cross-correlation.
The database directory should contain one image per student named <studentID>.<ext>.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from utils.image_processing import normalize_signature, image_similarity


SIG_TARGET_SIZE = (128, 64)    # (width, height) of normalized signature
MATCH_THRESHOLD = 0.45         # minimum NCC to accept a match
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _load_database(signatures_dir):
    """Load and normalize all reference signatures, returns {student_id: image}."""
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
    Identify a signature against the class database.
    Returns (best_id, best_score) — best_id is None if no match exceeds the threshold.
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
