# Système Intelligent d'Analyse et d'Extraction Automatique de Données

> Projet de fin d'année 
> Réalisé en binôme dans le cadre du module Deep Learning

---

## But du projet

Les entreprises traitent chaque jour des dizaines ou centaines de documents financiers (factures, reçus, bons de commande) de façon **manuelle**, ce qui engendre des erreurs, une perte de temps importante et des difficultés de centralisation des données.

Ce projet vise à concevoir un **système intelligent end-to-end** capable d'automatiser l'extraction structurée de données à partir de ces documents hétérogènes, en combinant :

- La **vision par ordinateur** (CNN / Deep Learning) pour détecter et classifier les documents
- L'**OCR** (Reconnaissance Optique de Caractères) pour extraire le texte des zones clés
- Le **traitement du langage naturel** (NLP / NER) pour structurer les entités extraites
- Une **interface web déployable** pour rendre le système utilisable immédiatement

**Résultat attendu :** un utilisateur uploade une facture ou un reçu → le système retourne automatiquement un JSON/CSV/Excel structuré contenant les champs clés (montant, date, fournisseur, total, etc.) en moins de 5 secondes.

---

## Architecture du projet

```
Système intelligent d'analyse et d'extraction automatique de données/
│
├── README.md                  ← Ce fichier
├── requirements.txt           ← Dépendances Python
├── Dockerfile                 ← Conteneurisation de l'application
├── docker-compose.yml         ← Orchestration des services
├── .gitignore                 ← Fichiers exclus du versionnement
│
├── notebooks/                 ← Expérimentation et prototypage
│   ├── 01_preprocessing.ipynb
│   ├── 02_classification_cnn.ipynb
│   ├── 03_ocr_extraction.ipynb
│   └── 04_pipeline_test.ipynb
│
├── src/                       ← Code source de production
│   ├── preprocessing/         ← Traitement des images en entrée
│   ├── classification/        ← Modèle CNN de classification
│   ├── ocr/                   ← Moteur OCR et post-processing
│   └── extraction/            ← NER et structuration des données
│
├── app/                       ← Application déployable
│   ├── main.py                ← API FastAPI (backend)
│   └── streamlit_app.py       ← Interface utilisateur (frontend)
│
├── data/                      ← Données (ignorées par Git)
│   ├── raw/                   ← Données brutes téléchargées
│   ├── processed/             ← Données après prétraitement
│   └── samples/               ← Exemples de documents pour les tests
│
└── models/                    ← Modèles entraînés (ignorés par Git)
```

---

## Description des répertoires

### `notebooks/`
Dossier dédié à l'**expérimentation** et à la recherche. Chaque notebook correspond à une étape du pipeline :
- `01_preprocessing` : exploration et validation du pipeline de prétraitement des images
- `02_classification_cnn` : entraînement et évaluation du modèle de classification (ResNet50 / EfficientNet)
- `03_ocr_extraction` : tests OCR, validation des expressions régulières, évaluation NER
- `04_pipeline_test` : test du pipeline complet de bout en bout sur des documents réels

### `src/`
Code Python **modulaire et réutilisable**, prêt pour la production. Chaque sous-dossier correspond à une brique du pipeline :

- **`preprocessing/`** : redimensionnement des images (224×224), normalisation, binarisation, correction d'inclinaison (deskewing), data augmentation (rotation, bruit, contraste)
- **`classification/`** : chargement du modèle CNN pré-entraîné, inférence du type de document (facture, reçu, bon de commande, relevé bancaire, contrat)
- **`ocr/`** : détection des zones d'intérêt (bounding boxes), extraction du texte par OCR, validation par expressions régulières (montants, dates, références)
- **`extraction/`** : reconnaissance d'entités nommées (NER), structuration en JSON, export CSV/Excel

### `app/`
Couche applicative exposant le pipeline sous deux formes :
- **`main.py`** : API REST (FastAPI) avec l'endpoint `POST /analyze` qui accepte un PDF ou une image et retourne un JSON structuré
- **`streamlit_app.py`** : interface web permettant d'uploader un document, visualiser les résultats annotés et télécharger l'export

### `data/`
Stockage local des datasets (non versionné sur GitHub car trop volumineux) :
- **`raw/`** : données brutes — SROIE Dataset, Invoices Kaggle, FUNSD
- **`processed/`** : données après preprocessing, prêtes pour l'entraînement
- **`samples/`** : quelques documents de test représentatifs pour les démonstrations

### `models/`
Fichiers des modèles entraînés (`.h5`, `.pt`, `.pkl`) — non versionnés sur GitHub. Les modèles finaux seront partagés via Google Drive ou Hugging Face Hub.

---

## Technologies utilisées

### Deep Learning & Vision

| Technologie | Rôle dans le projet |
|---|---|
| **TensorFlow / Keras** | Framework principal pour construire, entraîner et sauvegarder les modèles CNN (classification + détection de régions) |
| **ResNet50 / EfficientNetB3** | Architectures CNN pré-entraînées sur ImageNet utilisées en Transfer Learning pour classifier les types de documents |
| **OpenCV** | Bibliothèque de vision par ordinateur pour le prétraitement des images (redimensionnement, binarisation, deskewing, détection de contours) |

### OCR & NLP

| Technologie | Rôle dans le projet |
|---|---|
| **EasyOCR** | Moteur OCR principal, supporte le multi-langues (français, anglais, arabe). Utilisé pour extraire le texte des zones détectées dans les documents |
| **Tesseract + pytesseract** | Alternative OCR open-source de Google. Utilisé en fallback ou comparaison selon la qualité du document |
| **spaCy / Regex** | Post-processing NLP : validation des formats extraits (montants `\d+[.,]\d{2}`, dates, numéros de référence) et reconnaissance d'entités nommées |

### Backend & API

| Technologie | Rôle dans le projet |
|---|---|
| **FastAPI** | Framework Python moderne et rapide pour exposer le pipeline sous forme d'API REST. Gère l'upload de documents, déclenche le pipeline et retourne les résultats en JSON |
| **pdf2image** | Conversion de fichiers PDF en images avant traitement par le pipeline (un PDF → une liste d'images par page) |
| **pandas / openpyxl** | Structuration des données extraites et export en formats CSV et Excel pour les utilisateurs finaux |

### Frontend & Interface

| Technologie | Rôle dans le projet |
|---|---|
| **Streamlit** | Interface web Python pour la démonstration et l'utilisation du système. Permet l'upload de documents, la visualisation des bounding boxes annotées, et le téléchargement des résultats |

### Déploiement & DevOps

| Technologie | Rôle dans le projet |
|---|---|
| **Docker** | Conteneurisation de l'application complète (backend FastAPI + frontend Streamlit) pour garantir un déploiement reproductible sur n'importe quelle machine |
| **GitHub** | Versionnement du code source, collaboration en binôme via branches et pull requests |

### Datasets

| Dataset | Description |
|---|---|
| **SROIE Dataset** | 1 000 reçus annotés avec 4 champs clés : company, date, address, total. Référence du ICDAR 2019 Challenge |
| **Invoices Kaggle** | Factures variées multi-langues et multi-formats avec annotations de champs |
| **FUNSD** | 199 formulaires annotés, utile pour la segmentation et la détection de zones de texte |

---

## Pipeline général

```
[Document PDF/Image]
        ↓
[Preprocessing]        → Resize, normalisation, binarisation, deskewing
        ↓
[Classification CNN]   → Type de document détecté (facture, reçu, etc.)
        ↓
[Détection de zones]   → Bounding boxes sur montants, dates, fournisseur, total
        ↓
[OCR par zone]         → Extraction du texte brut par région
        ↓
[Post-processing NLP]  → Validation Regex + NER (entités nommées)
        ↓
[Export structuré]     → JSON / CSV / Excel
```

---

## Métriques d'évaluation

- **Classification :** Accuracy, Precision, Recall, F1-Score, Matrice de confusion
- **Détection de régions :** IoU (Intersection over Union), mAP
- **OCR :** CER (Character Error Rate), WER (Word Error Rate)
- **Extraction de champs :** Exact Match, Partial Match, F1 par champ
- **Business :** Temps de traitement < 5 secondes par document

---

## Installation rapide

```bash
# Cloner le repo
git clone https://github.com/othmaneboushabi/Syst-me-intelligent-d-analyse-et-d-extraction-automatique-de-donn-es.git

# Créer l'environnement virtuel
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux / macOS

# Installer les dépendances
pip install -r requirements.txt

# Lancer l'API
uvicorn app.main:app --reload

# Lancer l'interface
streamlit run app/streamlit_app.py
```

Ou avec Docker :

```bash
docker-compose up --build
```

---


> Projet encadré dans le cadre du module Deep Learning 