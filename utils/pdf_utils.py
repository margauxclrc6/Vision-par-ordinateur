"""PDF-to-image conversion using pdf2image (poppler backend)."""

import numpy as np
import cv2
from pdf2image import convert_from_path


PDF_DPI = 200    # higher = more accurate but slower


def pdf_to_images(pdf_path):
    """Convert every page of a PDF to a grayscale numpy array. Returns list of uint8 arrays."""
    pil_pages = convert_from_path(str(pdf_path), dpi=PDF_DPI)
    pages = []
    for pil_img in pil_pages:
        rgb = np.array(pil_img)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        pages.append(gray)
    return pages
