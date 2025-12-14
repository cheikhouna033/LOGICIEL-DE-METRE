# ============================================================
# LOGICIEL DE MÉTRÉ BTP – YOLOv8 + ÉCHELLE + VOLUMES + ACIER
# ============================================================

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import pandas as pd
from pdf2image import convert_from_bytes
import os

# ============================================================
# CONFIG STREAMLIT
# ============================================================
st.set_page_config(
    page_title="Logiciel de Métré BTP intelligent",
    layout="wide"
)

st.title("🏗️ Logiciel de métré automatique à partir de plans de coffrage")
st.caption("Vision artificielle • Métré béton • Estimation ferraillage")

# ============================================================
# CHEMIN MODÈLE YOLO (ADAPTE SI BESOIN)
# ============================================================
MODEL_PATH = r"C:\Users\PC\PycharmProjects\PythonProject1\runs\detect\metre_final7\weights\best.pt"

# ============================================================
# PARAMÈTRES MÉTIER BTP
# ============================================================
HAUTEURS_PAR_CLASSE = {
    "poutre": 0.50,
    "poteau": 3.00,
    "voile": 3.00,
    "dalle_pleine": 0.15,
    "radier": 0.25,
    "semelle_iso": 0.40,
    "semelle_filante": 0.40,
    "poutre_voile": 0.60,
}

RATIO_ACIER = {
    "poutre": 120,
    "poteau": 180,
    "voile": 110,
    "dalle_pleine": 80,
    "radier": 90,
    "semelle_iso": 70,
    "semelle_filante": 70,
    "poutre_voile": 130,
}

# ============================================================
# CHARGEMENT MODÈLE
# ============================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        st.error("❌ Fichier best.pt introuvable")
        st.stop()
    return YOLO(MODEL_PATH)

model = load_model()

# ============================================================
# FONCTIONS UTILITAIRES
# ============================================================
def load_image(uploaded_file):
    if uploaded_file.type == "application/pdf":
        st.error("❌ Les PDF ne sont pas supportés sur Streamlit Cloud. Veuillez importer une image.")
        st.stop()
    return Image.open(uploaded_file).convert("RGB")

def pil_to_cv(img):
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def cv_to_pil(img):
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# ============================================================
# SIDEBAR – IMPORT
# ============================================================
st.sidebar.header("📂 Import du plan")
uploaded_file = st.sidebar.file_uploader(
    "Plan de coffrage (image ou PDF)",
    type=["png", "jpg", "jpeg", "pdf"]
)

conf = st.sidebar.slider("Seuil de confiance YOLO", 0.1, 0.9, 0.4, 0.05)

if uploaded_file is None:
    st.info("⬅️ Importez un plan pour commencer")
    st.stop()

image_pil = load_image(uploaded_file)
image_cv = pil_to_cv(image_pil)

st.subheader("📐 Plan de coffrage")
st.image(image_pil, use_container_width=True)

# ============================================================
# ÉCHELLE DU PLAN (VERSION SIMPLE)
# ============================================================
st.subheader("📏 Échelle du plan")

echelle = st.selectbox(
    "Choisir l’échelle du plan",
    ["1/50", "1/75", "1/100", "1/150", "1/200"],
    index=2
)

DENOM = {
    "1/50": 50,
    "1/75": 75,
    "1/100": 100,
    "1/150": 150,
    "1/200": 200
}

scale_factor = 1 / DENOM[echelle]
st.success(f"Échelle définie : {echelle} → 1 px ≈ {scale_factor:.5f} m")

# ============================================================
# DÉTECTION YOLO
# ============================================================
st.subheader("🧠 Détection des éléments structurels")

results = model.predict(
    source=image_cv,
    conf=conf,
    verbose=False
)[0]

annotated = image_cv.copy()
elements = []

for box, cls_id, score in zip(
    results.boxes.xyxy.cpu().numpy(),
    results.boxes.cls.cpu().numpy(),
    results.boxes.conf.cpu().numpy()
):
    x1, y1, x2, y2 = map(int, box)
    label = results.names[int(cls_id)]

    cv2.rectangle(annotated, (x1, y1), (x2, y2), (0,255,0), 2)
    cv2.putText(
        annotated, label, (x1, y1-5),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1
    )

    largeur_m = (x2 - x1) * scale_factor
    longueur_m = (y2 - y1) * scale_factor
    hauteur = HAUTEURS_PAR_CLASSE.get(label, None)

    if hauteur:
        volume = largeur_m * longueur_m * hauteur
        acier = volume * RATIO_ACIER.get(label, 0)
    else:
        volume = acier = None

    elements.append({
        "Classe": label,
        "Confiance": round(float(score), 3),
        "Largeur (m)": round(largeur_m, 3),
        "Longueur (m)": round(longueur_m, 3),
        "Hauteur BTP (m)": hauteur,
        "Volume béton (m³)": round(volume, 3) if volume else None,
        "Acier estimé (kg)": round(acier, 1) if acier else None
    })

st.image(cv_to_pil(annotated), caption="Plan annoté", use_container_width=True)

# ============================================================
# TABLEAU FINAL
# ============================================================
st.subheader("📊 Tableau de métré")

df = pd.DataFrame(elements)
st.dataframe(df, use_container_width=True)

csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    "📥 Télécharger le métré (CSV)",
    csv,
    "metre_btp.csv",
    "text/csv"
)

st.caption("Métré automatique BTP basé sur la vision artificielle")
