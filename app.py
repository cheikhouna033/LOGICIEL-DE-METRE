# app.py
# --------------------------------------------------
# Application Streamlit : Détection + Métré + Déboursé
# --------------------------------------------------
import streamlit as st
from ultralytics import YOLO
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import pytesseract
import re
from pdf2image import convert_from_bytes

# -----------------------------
# CONFIG
# -----------------------------
st.set_page_config(page_title="Métré BTP Intelligent", layout="wide")

# Hypothèses BTP (prototype)
FACTEUR_ECHELLE = 0.01   # m / pixel (à adapter selon l'échelle du plan)
HAUTEUR_ETAGE = 3.0     # m
EPAISSEUR_VOILE = 0.20  # m
EPAISSEUR_DALLE = 0.15  # m

# Prix unitaires (exemple FCFA)
PRIX_UNITAIRES = {
    "poutre": {"ml": 750},
    "poteau": {"m3": 85000},
    "voile": {"m3": 90000},
    "dalle_pleine": {"m2": 4000},
    "dalle_hourdis": {"m2": 3500},
    "semelle_iso": {"m3": 90000},
    "semelle_filante": {"m3": 90000},
    "radier": {"m3": 95000},
}

# -----------------------------
# CHARGEMENT MODELE
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO(r"C:\Users\PC\PycharmProjects\PythonProject1\runs\detect\metre_final7\weights\best.pt")

model = load_model()

# -----------------------------
# UTILITAIRES
# -----------------------------

def pdf_to_image(uploaded_file):
    pages = convert_from_bytes(uploaded_file.read())
    return pages[0]


def load_image(uploaded_file):
    if uploaded_file.type == "application/pdf":
        return pdf_to_image(uploaded_file)
    return Image.open(uploaded_file).convert("RGB")


def calcul_metre(classe, w_px, h_px):
    w_m = w_px * FACTEUR_ECHELLE
    h_m = h_px * FACTEUR_ECHELLE

    if classe == "poutre":
        return w_m, "ml"
    elif classe == "poteau":
        return w_m * h_m * HAUTEUR_ETAGE, "m3"
    elif classe == "voile":
        return w_m * EPAISSEUR_VOILE * HAUTEUR_ETAGE, "m3"
    elif classe == "dalle_pleine" or "dalle_hourdis" in classe:
        return w_m * h_m, "m2"
    elif classe == "semelle_iso":
        return w_m * h_m * 0.5, "m3"
    elif classe == "semelle_filante":
        return w_m * 0.6 * 0.4, "m3"
    elif classe == "radier":
        return w_m * h_m * EPAISSEUR_DALLE, "m3"
    return 0, "-"


def calcul_debourse(classe, qte, unite):
    key = classe.split("_")[0]
    prix = PRIX_UNITAIRES.get(key, {})
    return qte * prix.get(unite, 0)


def preprocess_ocr(roi):
    gray = cv2.cvtColor(np.array(roi), cv2.COLOR_RGB2GRAY)
    return cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY,11,2)


def parse_dims(text):
    m = re.search(r"(\d{2,3})\s*[xX×]\s*(\d{2,3})", text)
    if m:
        return int(m.group(1))/100, int(m.group(2))/100
    return None

# -----------------------------
# INTERFACE
# -----------------------------
st.title("🏗️ Métré automatique à partir de plans de coffrage")

uploaded_file = st.file_uploader("Importer un plan (PDF / Image)", type=["pdf","png","jpg","jpeg"])
conf = st.slider("Seuil de confiance", 0.1, 0.9, 0.25, 0.05)
niveau = st.selectbox("Niveau", ["RDC","R+1","R+2","Sous-sol"])

if uploaded_file:
    image_pil = load_image(uploaded_file)
    img_np = np.array(image_pil)

    st.image(image_pil, caption="Plan importé", use_column_width=True)

    results = model.predict(img_np, conf=conf, verbose=False)[0]

    rows = []
    for box, cls_id in zip(results.boxes.xyxy, results.boxes.cls):
        x1,y1,x2,y2 = map(int, box)
        classe = results.names[int(cls_id)]

        w_px = x2-x1
        h_px = y2-y1

        qte, unite = calcul_metre(classe, w_px, h_px)
        cout = calcul_debourse(classe, qte, unite)

        rows.append({
            "Niveau": niveau,
            "Classe": classe,
            "Quantité": round(qte,2),
            "Unité": unite,
            "Coût FCFA": round(cout,0)
        })

    df = pd.DataFrame(rows)
    df_resume = df.groupby(["Niveau","Classe","Unité"], as_index=False).sum()

    st.subheader("📊 Pré-métré & Déboursé")
    st.dataframe(df_resume, use_container_width=True)

    st.download_button(
        "📥 Télécharger CSV",
        df_resume.to_csv(index=False).encode("utf-8"),
        "metre_debourse.csv",
        "text/csv"
    )