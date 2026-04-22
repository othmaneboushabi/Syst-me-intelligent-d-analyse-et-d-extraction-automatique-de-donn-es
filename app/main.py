import os
import cv2
import json
import re
import numpy as np
from pathlib import Path
from PIL import Image
import easyocr
import tensorflow as tf
from tensorflow.keras.models import load_model
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import uvicorn
import shutil
import tempfile

# ─────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────
IMG_SIZE    = (224, 224)
MODELS_DIR  = "../models"
CLASSES     = ["bon_de_commande", "contrat", "facture", "recu", "releve_bancaire"]

app = FastAPI(
    title="Document AI — Extraction automatique",
    description="API d'analyse et extraction de données depuis documents financiers",
    version="1.0.0"
)

# ─────────────────────────────────────────
# Chargement des modèles au démarrage
# ─────────────────────────────────────────
print("Chargement des modèles...")

cnn_model = load_model(f"{MODELS_DIR}/best_model_phase2.h5")
print("Modèle CNN chargé ✅")

ocr_reader = easyocr.Reader(['fr', 'en'], gpu=False)
print("EasyOCR chargé ✅")

# ─────────────────────────────────────────
# Fonctions utilitaires
# ─────────────────────────────────────────
def preprocess_image(img_path: str) -> np.ndarray:
    """Prépare une image pour la classification CNN."""
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0
    return np.expand_dims(img, axis=0)

def classify_document(img_path: str) -> dict:
    """Classifie le type de document."""
    img_array = preprocess_image(img_path)
    predictions = cnn_model.predict(img_array, verbose=0)
    class_idx   = np.argmax(predictions[0])
    confidence  = float(predictions[0][class_idx])
    return {
        "type":       CLASSES[class_idx],
        "confidence": round(confidence * 100, 2),
        "all_scores": {
            CLASSES[i]: round(float(predictions[0][i]) * 100, 2)
            for i in range(len(CLASSES))
        }
    }

def extract_fields(ocr_results: list) -> dict:
    """Extrait les champs clés depuis les résultats OCR."""
    full_text = " ".join([text for (_, text, conf) in ocr_results if conf > 0.3])
    lines     = [text for (_, text, conf) in ocr_results if conf > 0.3]

    fields = {
        "company": None,
        "date":    None,
        "total":   None,
        "address": None,
    }

    # Date
    date_patterns = [
        r'\d{2}/\d{2}/\d{4}',
        r'\d{2}-\d{2}-\d{4}',
        r'\d{4}-\d{2}-\d{2}',
        r'\d{2}\.\d{2}\.\d{4}',
    ]
    for pattern in date_patterns:
        match = re.search(pattern, full_text)
        if match:
            fields["date"] = match.group()
            break

    # Total
    total_patterns = [
        r'(?:total|montant|amount)[^\d]*(\d+[\.,]\d{2})',
        r'(\d+[\.,]\d{2})\s*(?:rm|€|\$|mad|dh)?$',
    ]
    for pattern in total_patterns:
        match = re.search(pattern, full_text, re.IGNORECASE)
        if match:
            fields["total"] = match.group(1)
            break

    # Company
    if lines:
        fields["company"] = lines[0].strip()

    # Address
    address_keywords = ['jalan', 'rue', 'avenue', 'boulevard',
                        'road', 'street', 'no.', 'lot', 'taman']
    address_lines = [
        line.strip() for line in lines
        if any(kw in line.lower() for kw in address_keywords)
    ]
    if address_lines:
        fields["address"] = ", ".join(address_lines[:2])

    return fields

# ─────────────────────────────────────────
# Routes API
# ─────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Document AI API",
        "version": "1.0.0",
        "endpoints": ["/analyze", "/health", "/docs"]
    }

@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": True}

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """
    Analyse un document financier :
    1. Classification du type de document
    2. Extraction OCR des champs clés
    3. Retourne un JSON structuré
    """
    # Vérifier le format
    allowed = {".jpg", ".jpeg", ".png", ".bmp"}
    ext = Path(file.filename).suffix.lower()
    if ext not in allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Format non supporté : {ext}. Utilisez : {allowed}"
        )

    # Sauvegarder temporairement
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        # 1. Classification
        classification = classify_document(tmp_path)

        # 2. OCR
        ocr_results = ocr_reader.readtext(tmp_path)

        # 3. Extraction des champs
        fields = extract_fields(ocr_results)

        # 4. Réponse finale
        response = {
            "filename":       file.filename,
            "document_type":  classification["type"],
            "confidence":     classification["confidence"],
            "extracted_fields": fields,
            "ocr_zones":      len(ocr_results),
            "all_scores":     classification["all_scores"]
        }

        return JSONResponse(content=response)

    finally:
        os.unlink(tmp_path)

# ─────────────────────────────────────────
# Lancement
# ─────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)