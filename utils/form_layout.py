"""
Layout descriptors for the exam form pages.
Coordinates are (x_ratio, y_ratio, w_ratio, h_ratio) as fractions of page width/height,
so they stay resolution-independent. Calibrate values against the actual form if needed.
"""

# Page 1 fields

PAGE1_FIELDS = {
    # CODES EXAM row — converted from reference canvas (2483×3510 px)
    "module":     (0.161, 0.091, 0.090, 0.015),
    "professor":  (0.390, 0.091, 0.090, 0.015),
    "date":       (0.621, 0.091, 0.093, 0.015),
    "code":       (0.848, 0.091, 0.090, 0.015),
    # Exam conditions checkboxes
    "notes_cours":       (0.04, 0.555, 0.08, 0.030),
    "notes_manuscrites": (0.19, 0.555, 0.08, 0.030),
    "ordinateur":        (0.35, 0.555, 0.08, 0.030),
    "calculatrice":      (0.51, 0.555, 0.08, 0.030),
    "feuilles_brouillon":     (0.66, 0.555, 0.08, 0.030),
    "feuilles_brouillon_nb":  (0.20, 0.585, 0.10, 0.025),
    # Note maximale / Note pour valider — converted from reference canvas
    "note_maximale": (0.545, 0.709, 0.121, 0.044),
    "note_valider":  (0.545, 0.757, 0.121, 0.044),
    # Prénom / Nom (handwritten boxes)
    "prenom": (0.03, 0.155, 0.32, 0.045),
    "nom":    (0.03, 0.215, 0.32, 0.045),
    # Signature zone — converted from reference canvas (258,984,736,422)
    "signature": (0.104, 0.280, 0.296, 0.120),
    # Group / StudentID — handled by grid_reader
    "group":      (0.37, 0.19, 0.24, 0.36),
    "student_id": (0.75, 0.19, 0.22, 0.36),
    # Cryptogram — converted from reference canvas (598,3378,98,98)
    "cryptogram": (0.241, 0.962, 0.039, 0.028),
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
