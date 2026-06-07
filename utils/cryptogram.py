"""
Cryptogram comparison utilities.
Checks that the small graphic at the bottom of every page matches page 1's cryptogram.
"""

import cv2
import numpy as np
from utils.image_processing import preprocess, normalize_signature, image_similarity
from utils.form_layout import PAGE1_FIELDS, crop_field


CRYPTO_THRESHOLD = 0.70
CRYPTO_SIZE = (64, 32)


def extract_cryptogram(page_gray):
    """Extract and normalize the cryptogram from the bottom of a page."""
    crop = crop_field(page_gray, PAGE1_FIELDS["cryptogram"])
    _, binary = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    resized = cv2.resize(binary, CRYPTO_SIZE, interpolation=cv2.INTER_AREA)
    return resized


def validate_cryptograms(pages_gray):
    """Check that all pages share the same cryptogram as page 1. Returns (valid, scores)."""
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
