"""
Ground-truth extraction from file names.

The provided data encodes the true student ID in each file name, e.g.
    EXAM_FORM1_63807.jpeg   -> student 63807   (presence photo)
    EXAM_FORM1_0001.pdf     -> (form index, not a student ID)

This lets us build an annotated evaluation set automatically, as required by
the methodology (Section 4.2): a labelled base for train / validation / test.
"""

import re
from pathlib import Path

# A student ID is a 5-digit number; form indices are 4-digit (0001..0068).
_STUDENT_ID_RE = re.compile(r"_(\d{5})(?:\b|_|\.)")


def student_id_from_name(filename):
    """
    Return the ground-truth student ID embedded in a file name, or None.
    e.g. 'EXAM_FORM1_63807.jpeg' -> '63807'
    """
    stem = Path(filename).stem
    m = _STUDENT_ID_RE.search(stem)
    return m.group(1) if m else None


def list_labelled_images(presences_dir, extensions=None):
    """
    Yield (path, ground_truth_id) for every labelled image in a presences dir.
    Skips files whose name does not encode a 5-digit student ID.
    """
    if extensions is None:
        extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
                      ".heic", ".heif", ".webp"}
    pairs = []
    for path in sorted(Path(presences_dir).iterdir()):
        if path.suffix.lower() not in extensions:
            continue
        gt = student_id_from_name(path.name)
        if gt is not None:
            pairs.append((path, gt))
    return pairs
