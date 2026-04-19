import os
import cv2
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm

# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
IMG_SIZE = (224, 224)       # Taille attendue par ResNet50
DATA_RAW_SROIE    = "data/raw/sroie/train/img"
DATA_RAW_INVOICES = "data/raw/invoices"
DATA_OUTPUT       = "data/processed"

# ─────────────────────────────────────────
# 1. Chargement et redimensionnement
# ─────────────────────────────────────────
def load_and_resize(image_path: str) -> np.ndarray:
    """Charge une image et la redimensionne en 224x224."""
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Image introuvable : {image_path}")
    img = cv2.resize(img, IMG_SIZE)
    return img

# ─────────────────────────────────────────
# 2. Binarisation (noir/blanc)
# ─────────────────────────────────────────
def binarize(img: np.ndarray) -> np.ndarray:
    """Convertit en niveaux de gris et applique un seuil Otsu."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

# ─────────────────────────────────────────
# 3. Deskewing (correction d'inclinaison)
# ─────────────────────────────────────────
def deskew(img: np.ndarray) -> np.ndarray:
    """Corrige l'inclinaison d'une image scannée."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    coords = np.column_stack(np.where(gray < 128))
    if len(coords) == 0:
        return img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = 90 + angle
    (h, w) = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h),
                             flags=cv2.INTER_CUBIC,
                             borderMode=cv2.BORDER_REPLICATE)
    return rotated

# ─────────────────────────────────────────
# 4. Normalisation (0 à 1)
# ─────────────────────────────────────────
def normalize(img: np.ndarray) -> np.ndarray:
    """Normalise les pixels entre 0 et 1."""
    return img.astype(np.float32) / 255.0

# ─────────────────────────────────────────
# 5. Pipeline complet sur une image
# ─────────────────────────────────────────
def preprocess_image(image_path: str) -> np.ndarray:
    """Applique le pipeline complet sur une image."""
    img = load_and_resize(image_path)
    img = deskew(img)
    img = normalize(img)
    return img

# ─────────────────────────────────────────
# 6. Traitement batch sur un dossier
# ─────────────────────────────────────────
def preprocess_folder(input_dir: str, output_dir: str, label: str):
    """
    Traite toutes les images d'un dossier.
    Sauvegarde les résultats dans output_dir/label/
    """
    input_path  = Path(input_dir)
    output_path = Path(output_dir) / label
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}
    images = [f for f in input_path.rglob("*") if f.suffix.lower() in extensions]

    print(f"[{label}] {len(images)} images trouvées dans {input_dir}")

    for img_path in tqdm(images, desc=f"Preprocessing {label}"):
        try:
            img = load_and_resize(str(img_path))
            img = deskew(img)
            # Sauvegarde en BGR (avant normalisation float)
            out_file = output_path / img_path.name
            cv2.imwrite(str(out_file), img)
        except Exception as e:
            print(f"Erreur sur {img_path.name} : {e}")

# ─────────────────────────────────────────
# 7. Split 70 / 15 / 15
# ─────────────────────────────────────────
def split_dataset(processed_dir: str, label: str):
    """
    Divise les images en train / val / test
    selon le ratio 70% / 15% / 15%.
    """
    source = Path(processed_dir) / label
    images = list(source.glob("*.jpg")) + list(source.glob("*.png"))

    if len(images) == 0:
        print(f"Aucune image trouvée dans {source}")
        return

    train, temp   = train_test_split(images, test_size=0.30, random_state=42)
    val,   test   = train_test_split(temp,   test_size=0.50, random_state=42)

    for split_name, split_files in [("train", train), ("val", val), ("test", test)]:
        dest = Path(processed_dir) / split_name / label
        dest.mkdir(parents=True, exist_ok=True)
        for f in split_files:
            shutil.copy(f, dest / f.name)

    print(f"[{label}] Split → train:{len(train)} | val:{len(val)} | test:{len(test)}")

# ─────────────────────────────────────────
# 8. Main — lancer tout le pipeline
# ─────────────────────────────────────────
if __name__ == "__main__":
    labels = {
        "recu":    DATA_RAW_SROIE,
        "facture": DATA_RAW_INVOICES + "/train",
    }

    print("=== Preprocessing des images ===")
    for label, input_dir in labels.items():
        if os.path.exists(input_dir):
            preprocess_folder(input_dir, DATA_OUTPUT, label)
            split_dataset(DATA_OUTPUT, label)
        else:
            print(f"Dossier introuvable : {input_dir} — ignoré")

    print("\n=== Preprocessing terminé ===")
    print(f"Données prêtes dans : {DATA_OUTPUT}/")