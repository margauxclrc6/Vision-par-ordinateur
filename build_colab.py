"""
Génère un notebook Colab 100% autonome (DeepForm_Colab_Standalone.ipynb).

Tout le code des modules est recréé via des cellules `%%writefile`, donc le
notebook ne dépend PAS de Git : l'utilisateur fournit seulement ses données
(zip uploadé ou Google Drive) et exécute les cellules de haut en bas.
"""

import json
from pathlib import Path

ROOT = Path(__file__).parent

# Ordre des modules (utils d'abord, puis programmes, puis évaluation).
MODULES = [
    "utils/__init__.py",
    "utils/image_processing.py",
    "utils/pdf_utils.py",
    "utils/ocr_reader.py",
    "utils/form_layout.py",
    "utils/grid_reader.py",
    "utils/checkbox_reader.py",
    "utils/cryptogram.py",
    "utils/exam_page_parser.py",
    "utils/signature_matcher.py",
    "utils/ground_truth.py",
    "autoValidPresences.py",
    "autoReadForm.py",
    "optimize_threshold.py",
    "evaluate.py",
]


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def writefile_cell(relpath):
    body = (ROOT / relpath).read_text()
    header = f"%%writefile {relpath}\n"
    return code(header + body)


cells = []

cells.append(md(
"""# DeepForm — Correction automatique d'examens semi-structurés
### Projet Computer Vision · IG.2405 – 2026

Ce notebook est **autonome** : tout le code est inclus ci-dessous, il n'y a
**aucun `git clone`** à faire. Vous fournissez uniquement vos données
(un fichier `.zip` ou un dossier Google Drive) puis vous exécutez les
cellules de haut en bas.

**Pipeline :**
- **Programme 1 — `autoValidPresences`** : à partir des photos de 1ʳᵉ page,
  lit le `StudentID` sur la grille à bulles (méthodes bas-niveau) et
  authentifie la signature (vérification 1:1) → `EXAM_FORMXX_PRESENCES.xlsx`.
- **Programme 2 — `autoReadForm`** : à partir des PDF scannés, lit la page
  d'identification (PAGE-01) et les réponses QCM/numériques (onglet EXAM)
  → un `.xlsx` par PDF.

**Contrainte cahier des charges (§4.1) :** les éléments **graphiques**
(grilles, cases à cocher, signatures, cryptogrammes) sont traités uniquement
par des méthodes **bas-niveau** (filtrage, Hough, morphologie, rotation).
Seuls les **textes** imprimés/manuscrits utilisent l'OCR / un réseau de
neurones.
"""))

cells.append(md("## Étape 1 — Dépendances système et Python"))
cells.append(code(
"""# Poppler (PDF→image), Tesseract (OCR), libheif (photos HEIC)
!apt-get -qq install -y poppler-utils tesseract-ocr tesseract-ocr-fra libheif1 >/dev/null
!pip -q install opencv-python-headless pdf2image pytesseract openpyxl pillow-heif numpy >/dev/null
print("Dépendances installées.")
"""))

cells.append(md(
"""## Étape 2 — Recréer l'arborescence du code (autonome, sans Git)

Chaque cellule `%%writefile` écrit un module sur le disque Colab. Exécutez-les
toutes une fois ; les imports fonctionneront ensuite normalement.
"""))
cells.append(code("import os\nos.makedirs('utils', exist_ok=True)\nprint('Dossier utils/ prêt.')"))

for m in MODULES:
    cells.append(writefile_cell(m))

cells.append(md(
"""## Étape 3 — Fournir les données

Deux options. **Choisissez-en une** et renseignez `DATA_ROOT`.

Structure attendue sous `DATA_ROOT` :
```
DATA_ROOT/
├── STUDENT_CLASS_SIGNATURES/      (base des signatures)
├── EXAM_FORM1_PRESENCES/          (photos 1ʳᵉ page)        ← Programme 1
├── EXAM_FORM1_PDF/                (PDF scannés, optionnel) ← Programme 2
├── EXAM_FORM2_PRESENCES/  ...
└── EXAM_FORM3_PRESENCES/  ...
```
"""))

cells.append(md("### Option A — Uploader un `.zip`"))
cells.append(code(
"""# Exécutez cette cellule, choisissez votre archive .zip, puis attendez.
from google.colab import files
import zipfile, os

up = files.upload()                       # sélectionnez votre .zip
zip_name = next(iter(up))
os.makedirs('/content/data', exist_ok=True)
with zipfile.ZipFile(zip_name) as z:
    z.extractall('/content/data')
DATA_ROOT = '/content/data'
print('Extrait dans', DATA_ROOT)
print(os.listdir(DATA_ROOT))
"""))

cells.append(md("### Option B — Google Drive"))
cells.append(code(
"""# from google.colab import drive
# drive.mount('/content/drive')
# DATA_ROOT = '/content/drive/MyDrive/PROJET_DEEPFORM'   # adaptez ce chemin
# import os; print(os.listdir(DATA_ROOT))
"""))

cells.append(md(
"""## Étape 4 — Localiser les dossiers

Si vos dossiers sont dans un sous-répertoire du zip, cette cellule le détecte
automatiquement (elle cherche `STUDENT_CLASS_SIGNATURES`).
"""))
cells.append(code(
"""from pathlib import Path

def find_data_root(root):
    root = Path(root)
    for p in [root, *root.rglob('*')]:
        if p.is_dir() and (p / 'STUDENT_CLASS_SIGNATURES').exists():
            return p
    return root

DATA_ROOT = find_data_root(DATA_ROOT)
SIGNATURES_DIR = str(Path(DATA_ROOT) / 'STUDENT_CLASS_SIGNATURES')
print('DATA_ROOT      =', DATA_ROOT)
print('SIGNATURES_DIR =', SIGNATURES_DIR)
print('Contenu        :', [p.name for p in Path(DATA_ROOT).iterdir()])
"""))

cells.append(md(
"""## Étape 5 — Programme 1 : validation des présences

Pour chaque formulaire disponible (`EXAM_FORMx_PRESENCES`), génère
`EXAM_FORMx_PRESENCES.xlsx` (colonnes `imageName`, `studentID_grid`,
`studentID_signature`).
"""))
cells.append(code(
"""from pathlib import Path
from autoValidPresences import autoValidPresences

RESULTS_DIR = '/content/RESULTS'
Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)

for pres in sorted(Path(DATA_ROOT).glob('EXAM_FORM*_PRESENCES')):
    if pres.is_dir():
        print('\\n==>', pres.name)
        autoValidPresences(str(pres), SIGNATURES_DIR, RESULTS_DIR)
"""))

cells.append(md(
"""## Étape 6 — Programme 2 : lecture automatique des formulaires PDF

Pour chaque `EXAM_FORMx_PDF`, génère un `.xlsx` par PDF (onglets `PAGE-01`
et `EXAM`). *(Peut être long : ~quelques secondes par page.)*
"""))
cells.append(code(
"""from autoReadForm import autoReadForm

for pdfdir in sorted(Path(DATA_ROOT).glob('EXAM_FORM*_PDF')):
    if pdfdir.is_dir():
        print('\\n==>', pdfdir.name)
        autoReadForm(str(pdfdir), SIGNATURES_DIR, RESULTS_DIR)
"""))

cells.append(md(
"""## Étape 7 — (Optionnel) Évaluation quantitative

Méthodologie §4.2 : apprentissage / validation / test sur FORM1 / FORM2 /
FORM3 avec vérité terrain lue depuis les noms de fichiers. Optimise le seuil
de décision des signatures et reporte la *balanced accuracy*.
"""))
cells.append(code(
"""from optimize_threshold import collect_scores, optimise_threshold, _report

train = collect_scores(str(Path(DATA_ROOT)/'EXAM_FORM1_PRESENCES'), SIGNATURES_DIR)
val   = collect_scores(str(Path(DATA_ROOT)/'EXAM_FORM2_PRESENCES'), SIGNATURES_DIR)
test  = collect_scores(str(Path(DATA_ROOT)/'EXAM_FORM3_PRESENCES'), SIGNATURES_DIR)

t_train, acc = optimise_threshold(train)
t_val,   _   = optimise_threshold(val)
operating_t = round((t_train + t_val) / 2, 2)
print(f'Seuil opérationnel sélectionné = {operating_t:.2f}')
for name, recs in [('TRAIN', train), ('VALIDATION', val), ('TEST', test)]:
    _report(name, recs, operating_t)
"""))

cells.append(md("## Étape 8 — Télécharger les résultats"))
cells.append(code(
"""import shutil
from google.colab import files
shutil.make_archive('/content/EXAM_RESULTS', 'zip', RESULTS_DIR)
files.download('/content/EXAM_RESULTS.zip')
"""))

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"provenance": []},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = ROOT / "DeepForm_Colab_Standalone.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1))
print("Écrit:", out, "—", len(cells), "cellules")
