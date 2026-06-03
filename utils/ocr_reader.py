"""
OCR utilities for printed and handwritten text.

Printed text  → Tesseract (pytesseract)  — high accuracy on clean documents.
Handwritten   → a lightweight CNN trained on EMNIST / custom data.

The CNN is loaded once (lazy) and reused across calls.
"""

import cv2
import numpy as np

try:
    import pytesseract
    _TESSERACT_AVAILABLE = True
except ImportError:
    _TESSERACT_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False


# ── Tesseract helpers ──────────────────────────────────────────────────────────

def read_printed_text(img_gray, config="--psm 6"):
    """
    Read printed text from a grayscale image crop using Tesseract.

    Parameters
    ----------
    img_gray : np.ndarray
    config   : str  — Tesseract page-segmentation mode and options.

    Returns
    -------
    text : str  (stripped)
    """
    if not _TESSERACT_AVAILABLE:
        return ""
    # Light preprocessing for better OCR
    _, binary = cv2.threshold(img_gray, 0, 255,
                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(binary, config=config)
    return text.strip()


def read_printed_date(img_gray):
    """Read a date field (DD/MM/YYYY)."""
    return read_printed_text(img_gray,
                             config="--psm 7 -c tessedit_char_whitelist=0123456789/")


def read_printed_field(img_gray):
    """Read a single-line printed field."""
    return read_printed_text(img_gray, config="--psm 7")


# ── Handwritten digit CNN ──────────────────────────────────────────────────────

class _DigitCNN(object if not _TORCH_AVAILABLE else nn.Module):
    """Minimal LeNet-5 style CNN for single handwritten digit recognition."""

    def __init__(self):
        if not _TORCH_AVAILABLE:
            return
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


_digit_model = None
_digit_model_path = "models/digit_cnn.pth"


def _get_digit_model():
    global _digit_model
    if _digit_model is not None:
        return _digit_model
    if not _TORCH_AVAILABLE:
        return None
    import torch
    model = _DigitCNN()
    try:
        state = torch.load(_digit_model_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        _digit_model = model
    except FileNotFoundError:
        # Model not yet trained; fall back to Tesseract
        _digit_model = None
    return _digit_model


def read_handwritten_digit(img_gray):
    """
    Recognize a single handwritten digit from a 28×28-compatible crop.

    Falls back to Tesseract if the CNN model is not available.

    Returns
    -------
    digit : int  (0-9) or -1 on failure.
    """
    model = _get_digit_model()
    if model is not None:
        import torch
        resized = cv2.resize(img_gray, (28, 28), interpolation=cv2.INTER_AREA)
        _, binary = cv2.threshold(resized, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        tensor = torch.tensor(binary, dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0
        with torch.no_grad():
            logits = model(tensor)
        return int(logits.argmax(dim=1).item())

    # Tesseract fallback
    if _TESSERACT_AVAILABLE:
        text = read_printed_text(img_gray,
                                  config="--psm 10 -c tessedit_char_whitelist=0123456789")
        try:
            return int(text.strip())
        except ValueError:
            return -1
    return -1


def read_handwritten_text(img_gray):
    """
    Read free-form handwritten text (e.g. name, first name).
    Uses Tesseract with handwriting-friendly settings.
    """
    if not _TESSERACT_AVAILABLE:
        return ""
    # Scale up for better recognition
    h, w = img_gray.shape
    scale = max(1, 60 // h)
    resized = cv2.resize(img_gray, (w * scale, h * scale),
                         interpolation=cv2.INTER_CUBIC)
    return read_printed_text(resized, config="--psm 7")


def read_handwritten_number(img_gray):
    """
    Read a handwritten number (mantissa + exponent or integer).
    Returns the raw string.
    """
    if not _TESSERACT_AVAILABLE:
        return ""
    _, binary = cv2.threshold(img_gray, 0, 255,
                               cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    text = pytesseract.image_to_string(
        binary,
        config="--psm 7 -c tessedit_char_whitelist=0123456789.,-Ee "
    )
    return text.strip()
