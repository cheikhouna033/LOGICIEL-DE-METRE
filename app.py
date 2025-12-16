import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import pandas as pd
import math

st.set_page_config(layout="wide")
st.title("📐 Définition de l’échelle – Métré BTP")

# ======================
# IMPORT DU PLAN
# ======================
uploaded_file = st.file_uploader(
    "📂 Importer un plan (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is None:
    st.info("Veuillez importer un plan.")
    st.stop()

image = Image.open(uploaded_file).convert("RGB")
width, height = image.size

# ======================
# ÉTAPE 1 : TRAIT ÉCHELLE
# ======================
st.subheader("1️⃣ Tracer un trait de référence")

st.info(
    "Tracez un trait sur une longueur connue du plan "
    "(ex : une poutre ou une travée), puis saisissez sa longueur réelle."
)

canvas_scale = st_canvas(
    stroke_width=3,
    stroke_color="#FF0000",
    background_image=image,
    update_streamlit=True,
    width=width,
    height=height,
    drawing_mode="line",
    key="scale_canvas",
)

scale_factor = None

if canvas_scale.json_data is not None:
    lines = [
        obj for obj in canvas_scale.json_data["objects"]
        if obj["type"] == "line"
    ]

    if len(lines) > 0:
        line = lines[0]

        x1, y1 = line["x1"], line["y1"]
        x2, y2 = line["x2"], line["y2"]

        length_px = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

        st.write(f"📏 Longueur mesurée sur le plan : **{length_px:.1f} px**")

        real_length = st.number_input(
            "Entrer la longueur réelle correspondante (en mètres)",
            min_value=0.01,
            step=0.1
        )

        if real_length > 0:
            scale_factor = real_length / length_px
            st.success(
                f"✅ Échelle définie : **1 pixel = {scale_factor:.5f} m**"
            )

# ======================
# STOCKAGE DE L’ÉCHELLE
# ======================
if scale_factor:
    st.session_state["scale_factor"] = scale_factor
