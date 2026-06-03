"""
Cryptogram comparison utilities.

The cryptogram is the small graphic at the bottom of each page.
All pages of the same exam must have the same cryptogram.

Strategy:
  1. Extract the cryptogram region (low-level morphological crop).
  2. Normalize to a fixed size.
  3. Compare successive pages using NCC — all pages must match page 1.
"""

import cv2
import numpy as np
from utils.image_processing import preprocess, normalize_signature, image_similarity
from utils.form_layout import PAGE1_FIELDS, crop_field


# NCC threshold to consider two cryptograms identical
CRYPTO_THRESHOLD = 0.70
CRYPTO_SIZE = (64, 32)


def extract_cryptogram(page_gray):
    """Extract and normalize the cryptogram from the bottom of a page."""
    crop = crop_field(page_gray, PAGE1_FIELDS["cryptogram"])
    _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized = cv2.resize(binary, CRYPTO_SIZE, interpolation=cv2.INTER_AREA)
    return resized


def validate_cryptograms(pages_gray):
    """
    Verify that all pages share the same cryptogram as page 1.

    Returns
    -------
    valid : bool
    scores : list of float  (NCC of each page vs page 1)
    """
    if len(pages_gray) == 0:
        return False, []

    ref = extract_cryptogram(pages_gray[0])
    scores = [1.0]

    for page in pages_gray[1:]:
        crypto = extract_cryptogram(page)
        score = image_similarity(ref, crypto)
        scores.append(score)

    valid = all(s >= CRYPTO_THRESHOLD for s in scores[1:])
    return valid, scores
