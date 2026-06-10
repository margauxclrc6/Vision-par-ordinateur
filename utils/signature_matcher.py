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
MATCH_THRESHOLD = 0.15         # minimum NCC to accept a match
SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def _load_database(signatures_dir):
    """
    Load reference signatures recursively. Supports nested structure:
      signatures_dir/
        <studentID>/          ← folder named by student ID (numeric)
          <studentID>_000.png
          ...
        <scan_folder>/        ← any named folder
          <studentID>/
            ...
    Returns {student_id: [normalized_img, ...]}
    """
    db = {}

    def _walk(folder, current_id):
        for item in Path(folder).iterdir():
            if item.is_dir():
                sid = item.name if item.name.isdigit() else current_id
                _walk(item, sid)
            elif item.suffix.lower() in SUPPORTED_EXT and current_id:
                img = cv2.imread(str(item), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    norm = normalize_signature(img, SIG_TARGET_SIZE)
                    db.setdefault(current_id, []).append(norm)

    _walk(signatures_dir, None)
    return db


def match_signature(sig_gray, signatures_dir):
    """
    Identify a signature against the class database.
    Returns (best_id, best_score) — best_id is None if no match exceeds threshold.
    Compares the query against all samples per student and takes the max score.
    """
    db = _load_database(signatures_dir)
    if not db:
        return None, 0.0

    query = normalize_signature(sig_gray, SIG_TARGET_SIZE)

    best_id = None
    best_score = -1.0

    for student_id, refs in db.items():
        score = max(image_similarity(query, ref) for ref in refs)
        if score > best_score:
            best_score = score
            best_id = student_id

    if best_score < MATCH_THRESHOLD:
        return None, best_score

    return best_id, best_score
