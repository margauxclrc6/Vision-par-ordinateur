"""
Layout descriptors for the exam form pages.
Coordinates are (x_ratio, y_ratio, w_ratio, h_ratio) as fractions of page width/height,
so they stay resolution-independent. Calibrate values against the actual form if needed.
"""

# Page 1 fields

PAGE1_FIELDS = {
    # row 1 – Module
    # row 1 – Module (printed, top header band)
    "module":     (0.08, 0.055, 0.18, 0.025),
    # row 2 – Professor
    "professor":  (0.35, 0.055, 0.18, 0.025),
    # row 3 – Date
    "date":       (0.57, 0.055, 0.22, 0.025),
    # row 4 – Code
    "code":       (0.78, 0.055, 0.20, 0.025),
    # row 5 – Notes de cours YES checkbox
    "notes_cours":       (0.05, 0.615, 0.07, 0.025),
    # row 6 – Notes manuscrites YES checkbox
    "notes_manuscrites": (0.20, 0.615, 0.07, 0.025),
    # row 7 – Ordinateur portable YES checkbox
    "ordinateur":        (0.36, 0.615, 0.07, 0.025),
    # row 8 – Calculatrice YES checkbox
    "calculatrice":      (0.52, 0.615, 0.07, 0.025),
    # row 9 – Feuilles brouillon YES checkbox
    "feuilles_brouillon":     (0.68, 0.615, 0.07, 0.025),
    "feuilles_brouillon_nb":  (0.20, 0.645, 0.10, 0.025),
    # row 10 – Note maximale
    "note_maximale": (0.55, 0.71, 0.15, 0.04),
    # row 11 – Note pour valider
    "note_valider":  (0.55, 0.76, 0.15, 0.04),
    # row 13 – Prénom (handwritten boxes)
    "prenom": (0.03, 0.155, 0.32, 0.045),
    # row 14 – Nom (handwritten boxes)
    "nom":    (0.03, 0.215, 0.32, 0.045),
    # row 15 – Signature zone
    "signature": (0.03, 0.255, 0.30, 0.245),
    # row 16 – Group (from bubble grid)
    "group":     (0.37, 0.19, 0.24, 0.36),
    # row 17 – StudentID (from bubble grid)
    "student_id": (0.62, 0.19, 0.35, 0.36),
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
