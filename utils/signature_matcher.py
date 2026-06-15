"""
Matches a signature against the class database using zero-mean NCC.
The database directory contains one folder per student named <studentID>,
with one or more signature images inside.
"""

from pathlib import Path
import cv2
import numpy as np
from utils.image_processing import load_image, normalize_signature, ncc_similarity


SIG_TARGET_SIZE = (128, 64)    # (width, height) of normalized signature

# With zero-mean NCC, a real match typically scores > 0.35.
# Set conservatively so unrecognized signatures return None (column C empty)
# rather than a wrong ID.
MATCH_THRESHOLD = 0.35

SUPPORTED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
                 ".heic", ".heif", ".webp"}


def _load_database(signatures_dir):
    """
    Load all reference signatures, recursively.
    Supports:
      signatures_dir/<studentID>/<any_name>.jpg   (nested structure)
      signatures_dir/<studentID>.jpg              (flat structure)
    Returns {student_id: [normalized_img, ...]}
    """
    db = {}

    def _add(path, student_id):
        try:
            _, img = load_image(path)   # uses full fallback including HEIC
        except Exception:
            return
        norm = normalize_signature(img, SIG_TARGET_SIZE)
        if np.sum(norm) > 0:
            db.setdefault(student_id, []).append(norm)

    def _walk(folder, current_id):
        for item in sorted(Path(folder).iterdir()):
            if item.is_dir():
                # Directory name = student ID if numeric
                sid = item.name if item.name.isdigit() else current_id
                _walk(item, sid)
            elif item.suffix.lower() in SUPPORTED_EXT:
                # Flat structure: filename without extension = student ID
                sid = current_id or item.stem
                if sid:
                    _add(item, sid)

    _walk(signatures_dir, None)
    return db


_db_cache = {}   # {str(signatures_dir): db}


def match_signature(sig_gray, signatures_dir):
    """
    Identify a signature against the class database.
    Returns (best_id, best_score).
    best_id is None if score < MATCH_THRESHOLD (signature not recognised).
    DB is cached per directory.
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

    best_id = None
    best_score = -1.0

    for student_id, refs in db.items():
        score = max(ncc_similarity(query, ref) for ref in refs)
        if score > best_score:
            best_score = score
            best_id = student_id

    if best_score < MATCH_THRESHOLD:
        return None, best_score

    return best_id, best_score
