"""Run DeepForm locally from VS Code. Edit the paths below then: python run_local.py"""

import sys
from pathlib import Path

# ── À MODIFIER ──────────────────────────────────────────────────────────────
DATA_ROOT      = r"C:\Users\TON_NOM\Downloads\PROJECT 2026 -DATABASE-20260603"
RESULTS_ROOT   = r"C:\Users\TON_NOM\Downloads\DeepForm_Results"
FORMS          = ["FORM1", "FORM2", "FORM3"]
# ────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))

from autoValidPresences import autoValidPresences
from autoReadForm import autoReadForm

data = Path(DATA_ROOT)
results = Path(RESULTS_ROOT)

# Construire les dossiers attendus
presences_dirs = {}
pdf_dirs = {}
for form in FORMS:
    src = data / form
    if not src.exists():
        print(f"[SKIP] {src} introuvable")
        continue
    presences_dirs[form] = src
    pdf_dirs[form] = src

sig_dir = data / "SIGNATURES"
extracted_sigs = results / "SIGNATURES"
extracted_sigs.mkdir(parents=True, exist_ok=True)

# Extraire les signatures ZIP
import zipfile
if sig_dir.exists():
    for z in sig_dir.glob("*.zip"):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(extracted_sigs)
    print(f"Signatures extraites : {len(list(extracted_sigs.iterdir()))} fichiers")

# Lancer les deux programmes pour chaque formulaire
for form in FORMS:
    src = data / form
    if not src.exists():
        continue

    form_results = results / form
    form_results.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  {form} — Programme 1 : Validation des présences")
    print(f"{'='*50}")
    autoValidPresences(str(src), str(extracted_sigs), str(form_results))

    print(f"\n{'='*50}")
    print(f"  {form} — Programme 2 : Lecture des formulaires")
    print(f"{'='*50}")
    autoReadForm(str(src), str(extracted_sigs), str(form_results))

print(f"\nRésultats dans : {results}")
