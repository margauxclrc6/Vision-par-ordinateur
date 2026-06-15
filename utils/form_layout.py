"""
Layout descriptors for the exam form pages.
Coordinates are (x_ratio, y_ratio, w_ratio, h_ratio) as fractions of page width/height,
so they stay resolution-independent. Calibrate values against the actual form if needed.
"""

# Page 1 fields

PAGE1_FIELDS = {
    # CODES EXAM row — calibrated from PDF at 250 DPI
    # Module value "IG.1103" at x_ratio 0.193-0.290, y_ratio 0.108-0.125
    "module":     (0.193, 0.108, 0.10, 0.017),
    "professor":  (0.387, 0.108, 0.10, 0.017),
    "date":       (0.604, 0.108, 0.10, 0.017),
    "code":       (0.822, 0.108, 0.10, 0.017),
    # Exam conditions checkboxes — in the "Are authorised" section
    "notes_cours":       (0.04, 0.555, 0.08, 0.030),
    "notes_manuscrites": (0.19, 0.555, 0.08, 0.030),
    "ordinateur":        (0.35, 0.555, 0.08, 0.030),
    "calculatrice":      (0.51, 0.555, 0.08, 0.030),
    "feuilles_brouillon":     (0.66, 0.555, 0.08, 0.030),
    "feuilles_brouillon_nb":  (0.20, 0.585, 0.10, 0.025),
    # Note maximale "10" / Note pour valider "04" — calibrated from PDF
    "note_maximale": (0.556, 0.700, 0.145, 0.042),
    "note_valider":  (0.556, 0.742, 0.145, 0.028),
    # row 13 – Prénom (handwritten boxes)
    "prenom": (0.03, 0.155, 0.32, 0.045),
    # row 14 – Nom (handwritten boxes)
    "nom":    (0.03, 0.215, 0.32, 0.045),
    # row 15 – Signature zone
    "signature": (0.03, 0.255, 0.30, 0.245),
    # row 16 – Group (from bubble grid)
    "group":     (0.37, 0.19, 0.24, 0.36),
    # row 17 – StudentID (from bubble grid)
    "student_id": (0.75, 0.19, 0.22, 0.36),
    # row 18 – Cryptogram (bottom left)
    "cryptogram": (0.08, 0.935, 0.10, 0.05),
}

# Exam page fields

# Number of answer choices per question (A-H = 8 max, adapt to the actual form)
N_CHOICES = 8
CHOICE_LABELS = list("ABCDEFGH")[:N_CHOICES]

# Relative position of the answer grid on an exam page
EXAM_QUESTION_COL = (0.02, 0.10, 0.08, 0.80)
EXAM_CHOICES_START_X = 0.12
EXAM_CHOICE_COL_W = 0.07
EXAM_ROW_START_Y = 0.10
EXAM_ROW_H = 0.06

# Handwritten numerical answer columns
EXAM_MANTISSE_COL  = (0.70, 0.10, 0.15, 0.80)
EXAM_EXPOSANT_COL  = (0.86, 0.10, 0.08, 0.80)
EXAM_UNITE_COL     = (0.95, 0.10, 0.04, 0.80)


def field_to_pixels(rel_coords, page_h, page_w):
    """Convert relative (x,y,w,h) to integer pixel coordinates."""
    xr, yr, wr, hr = rel_coords
    return (int(xr * page_w), int(yr * page_h),
            int(wr * page_w), int(hr * page_h))


def crop_field(page_gray, rel_coords):
    """Crop a field from a page using relative coordinates."""
    h, w = page_gray.shape
    x, y, fw, fh = field_to_pixels(rel_coords, h, w)
    return page_gray[y:y+fh, x:x+fw]
