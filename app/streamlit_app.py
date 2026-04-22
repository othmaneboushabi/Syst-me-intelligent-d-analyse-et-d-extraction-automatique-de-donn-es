import streamlit as st
import requests
import json
import pandas as pd
from pathlib import Path
from PIL import Image
import io

# ─────────────────────────────────────────
# Configuration de la page
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Document AI — Extraction automatique",
    page_icon="📄",
    layout="wide"
)

API_URL = "http://localhost:8000"

# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────
st.title("📄 Système Intelligent d'Analyse de Documents")
st.markdown("Uploadez un document financier — le système le classifie et extrait automatiquement les données clés.")
st.divider()

# ─────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Informations")
    st.markdown("""
    **Documents supportés :**
    - Factures
    - Reçus
    - Bons de commande
    - Relevés bancaires
    - Contrats

    **Formats acceptés :**
    - JPG / JPEG
    - PNG
    - BMP

    **Pipeline :**
    1. Preprocessing image
    2. Classification CNN
    3. OCR EasyOCR
    4. Extraction NER
    5. Export JSON/CSV
    """)

    st.divider()

    # Statut API
    st.subheader("Statut API")
    try:
        response = requests.get(f"{API_URL}/health", timeout=3)
        if response.status_code == 200:
            st.success("API connectée ✅")
        else:
            st.error("API non disponible ❌")
    except:
        st.error("API non disponible ❌")
        st.info("Lancez d'abord : `uvicorn main:app --reload`")

# ─────────────────────────────────────────
# Zone d'upload
# ─────────────────────────────────────────
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Upload du document")
    uploaded_file = st.file_uploader(
        "Choisissez un document financier",
        type=["jpg", "jpeg", "png", "bmp"],
        help="Formats supportés : JPG, PNG, BMP"
    )

    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption=f"Document : {uploaded_file.name}", use_column_width=True)

with col2:
    st.subheader("🔍 Résultats d'analyse")

    if uploaded_file:
        with st.spinner("Analyse en cours..."):
            try:
                # Appel API
                files    = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(f"{API_URL}/analyze", files=files, timeout=60)

                if response.status_code == 200:
                    result = response.json()

                    # Type de document
                    doc_type   = result["document_type"].replace("_", " ").title()
                    confidence = result["confidence"]

                    st.success(f"Analyse terminée !")

                    # Métriques principales
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Type détecté", doc_type)
                    m2.metric("Confiance", f"{confidence:.1f}%")
                    m3.metric("Zones OCR", result["ocr_zones"])

                    st.divider()

                    # Champs extraits
                    st.subheader("📋 Champs extraits")
                    fields = result["extracted_fields"]

                    f1, f2 = st.columns(2)
                    with f1:
                        st.text_input("Société",        value=fields.get("company")  or "Non détecté", disabled=True)
                        st.text_input("Date",           value=fields.get("date")     or "Non détecté", disabled=True)
                    with f2:
                        st.text_input("Montant total",  value=fields.get("total")    or "Non détecté", disabled=True)
                        st.text_input("Adresse",        value=fields.get("address")  or "Non détecté", disabled=True)

                    st.divider()

                    # Scores par classe
                    st.subheader("📊 Scores de classification")
                    scores_df = pd.DataFrame([
                        {"Classe": k.replace("_", " ").title(), "Score (%)": v}
                        for k, v in result["all_scores"].items()
                    ]).sort_values("Score (%)", ascending=False)

                    st.bar_chart(scores_df.set_index("Classe"))

                    st.divider()

                    # Export JSON
                    st.subheader("💾 Export")
                    col_json, col_csv = st.columns(2)

                    with col_json:
                        json_str = json.dumps(result, indent=2, ensure_ascii=False)
                        st.download_button(
                            label="⬇️ Télécharger JSON",
                            data=json_str,
                            file_name=f"{Path(uploaded_file.name).stem}_result.json",
                            mime="application/json"
                        )

                    with col_csv:
                        csv_data = pd.DataFrame([{
                            "Fichier":        uploaded_file.name,
                            "Type":           result["document_type"],
                            "Confiance (%)":  result["confidence"],
                            "Société":        fields.get("company"),
                            "Date":           fields.get("date"),
                            "Total":          fields.get("total"),
                            "Adresse":        fields.get("address"),
                        }])
                        st.download_button(
                            label="⬇️ Télécharger CSV",
                            data=csv_data.to_csv(index=False, encoding='utf-8-sig'),
                            file_name=f"{Path(uploaded_file.name).stem}_result.csv",
                            mime="text/csv"
                        )

                    # JSON brut
                    with st.expander("Voir le JSON complet"):
                        st.json(result)

                else:
                    st.error(f"Erreur API : {response.status_code}")

            except requests.exceptions.ConnectionError:
                st.error("Impossible de contacter l'API. Vérifiez qu'elle est lancée.")
            except Exception as e:
                st.error(f"Erreur : {str(e)}")

    else:
        st.info("👆 Uploadez un document pour commencer l'analyse.")

# ─────────────────────────────────────────
# Footer
# ─────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:gray; font-size:12px;'>"
    "Document AI — EMSI IIR-IA · Deep Learning Project"
    "</div>",
    unsafe_allow_html=True
)