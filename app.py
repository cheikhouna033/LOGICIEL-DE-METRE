import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
from streamlit_drawable_canvas import st_canvas

# -------------------------------
# CONFIG
# -------------------------------
st.set_page_config(page_title="Logiciel de Métré BTP", layout="wide")

# -------------------------------
# TITRE
# -------------------------------
st.title("📐 Logiciel Intelligent de Métré BTP")
st.markdown("""
### Version Cloud – Démonstrateur académique

🔹 Import d’images de plans  
🔹 Annotation manuelle  
🔹 Définition d’échelle simple  
🔹 Calcul automatique des quantités  

⚠️ Les fichiers PDF sont traités **uniquement en version locale**.
""")

# -------------------------------
# UPLOAD IMAGE (PAS PDF)
# -------------------------------
uploaded_file = st.file_uploader(
    "📂 Charger un plan (image uniquement)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is None:
    st.info("⬆️ Veuillez charger une image de plan pour commencer.")
    st.stop()

image = Image.open(uploaded_file)

# -------------------------------
# AFFICHAGE IMAGE
# -------------------------------
st.subheader("🖼️ Plan chargé")
st.image(image, use_column_width=True)

# -------------------------------
# ÉCHELLE SIMPLE
# -------------------------------
st.subheader("📏 Définition de l’échelle")

col1, col2 = st.columns(2)

with col1:
    pixel_ref = st.number_input(
        "Longueur mesurée sur le plan (pixels)",
        min_value=1.0,
        value=100.0
    )

with col2:
    real_ref = st.number_input(
        "Longueur réelle correspondante (mètres)",
        min_value=0.01,
        value=1.0
    )

scale = real_ref / pixel_ref
st.success(f"✅ Échelle : **1 pixel = {scale:.4f} m**")

# -------------------------------
# CANVAS
# -------------------------------
st.subheader("✏️ Annotation du plan")

st.markdown("""
- 🟦 Rectangle → surfaces (dalles, radiers, voiles)  
- ➖ Ligne → éléments linéaires (poutres, semelles)
""")

st.subheader("✏️ Zone d’annotation (dessin libre)")

canvas = st_canvas(
    fill_color="rgba(0, 0, 255, 0.3)",
    stroke_width=2,
    stroke_color="#FF0000",
    background_color="#FFFFFF",
    update_streamlit=True,
    height=600,
    drawing_mode="rect",
    key="canvas",
)


# -------------------------------
# MÉTRÉ
# -------------------------------
results = []

if canvas.json_data and "objects" in canvas.json_data:
    for obj in canvas.json_data["objects"]:

        if obj["type"] == "rect":
            w_px = obj["width"]
            h_px = obj["height"]

            surface = (w_px * scale) * (h_px * scale)

            results.append({
                "Type": "Surface (dalle / radier / voile)",
                "Surface (m²)": round(surface, 2),
                "Longueur (m)": None
            })

        if obj["type"] == "line":
            x1, y1, x2, y2 = obj["x1"], obj["y1"], obj["x2"], obj["y2"]
            length_px = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            length_m = length_px * scale

            results.append({
                "Type": "Linéaire (poutre / semelle)",
                "Surface (m²)": None,
                "Longueur (m)": round(length_m, 2)
            })

# -------------------------------
# TABLEAU
# -------------------------------
if results:
    st.subheader("📊 Tableau de métré")

    df = pd.DataFrame(results)
    st.dataframe(df, use_container_width=True)

    st.subheader("📐 Totaux")
    st.metric("Surface totale (m²)", round(df["Surface (m²)"].dropna().sum(), 2))
    st.metric("Longueur totale (m)", round(df["Longueur (m)"].dropna().sum(), 2))
else:
    st.info("✏️ Dessinez des éléments pour afficher le métré.")
