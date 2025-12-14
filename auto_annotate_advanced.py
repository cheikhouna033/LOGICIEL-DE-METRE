import os
import cv2
import numpy as np
import pytesseract
from pdf2image import convert_from_path

# ========== CONFIG GÉNÉRALE ==========

# Dossiers
IMAGE_DIR = "data/images/train"
LABEL_DIR = "data/labels/train"
DEBUG_DIR = "debug_annotations"

os.makedirs(LABEL_DIR, exist_ok=True)
os.makedirs(DEBUG_DIR, exist_ok=True)

# Tesseract (ADAPTE ce chemin à ton installation)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Si tu veux lire directement des PDF dans IMAGE_DIR
POPPLER_PATH = r"C:\poppler-25.12.0\Library\bin"  # adapte à ta machine

# On ignore la légende à droite (20% de la largeur)
IGNORE_RIGHT_RATIO = 0.80

# Mapping des classes YOLO
CLASS_NAMES = {
    0: "poutre",
    1: "poteau",
    2: "voile",
    3: "dalle_pleine",
    4: "dalle_hourdis",
    5: "semelle_isolee",
    6: "semelle_filante",
    7: "poutre_voile",
    8: "radier",
}


# ========== FONCTIONS UTILITAIRES ==========

def load_image_any(path):
    """
    Charge une image ou la première page d'un PDF en OpenCV (BGR).
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        pages = convert_from_path(path, dpi=300, poppler_path=POPPLER_PATH)
        pil_img = pages[0].convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    else:
        img = cv2.imread(path)
    return img


def enhance_image_for_shapes(gray):
    """
    Améliore le contraste et binarise pour les lignes fines de plans.
    """
    # Egalisation locale du contraste
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(gray)

    # Seuillage adaptatif
    thr = cv2.adaptiveThreshold(
        cl, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        41, 10
    )

    # Petit closing pour fermer les trous
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    closed = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel, iterations=1)
    return closed


def normalize_xyxy(x1, y1, x2, y2, w_img, h_img):
    xc = (x1 + x2) / 2 / w_img
    yc = (y1 + y2) / 2 / h_img
    w = (x2 - x1) / w_img
    h = (y2 - y1) / h_img
    return xc, yc, w, h


def ocr_text(img, x1, y1, x2, y2, pad_ratio=0.2):
    """
    OCR dans une zone autour du rectangle.
    """
    h_img, w_img = img.shape[:2]
    w = x2 - x1
    h = y2 - y1
    dx = int(w * pad_ratio)
    dy = int(h * pad_ratio)

    x1p = max(x1 - dx, 0)
    y1p = max(y1 - dy, 0)
    x2p = min(x2 + dx, w_img)
    y2p = min(y2 + dy, h_img)

    roi = img[y1p:y2p, x1p:x2p]
    if roi.size == 0:
        return ""

    roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    roi_thr = cv2.threshold(roi_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    text = pytesseract.image_to_string(roi_thr, lang="fra+eng")
    text = text.replace("\n", " ").replace("\x0c", " ").strip().upper()
    return text


def heuristique_classe_geom_ocr(w, h, area, aspect, x, y, w_img, h_img, ocr):
    """
    Heuristique de classification BTP :
    - basée sur la géométrie + OCR simplifiée
    - renvoie un ID de classe (0..8) ou None
    """

    # Aide via OCR
    if "RADIER" in ocr:
        return 8
    if "SEMELLE" in ocr and "FIL" in ocr:
        return 6
    if "SEMELLE" in ocr:
        return 5
    if "VOILE" in ocr or "V1" in ocr or "V2" in ocr:
        return 2
    if "P." in ocr or "P " in ocr or "P(" in ocr:
        # souvent des poutres "P1(20X40)" etc.
        if area > 80000 and aspect > 2.5:
            return 0
        else:
            return 1  # poteau ou petit P

    # Poteau : petite forme quasi carrée
    if 40 < w < 200 and 40 < h < 200 and 0.8 < aspect < 1.3:
        return 1  # poteau

    # Poutre horizontale : rectangle très allongé
    if w > 220 and h < 120 and aspect > 3.0:
        return 0  # poutre

    # Voile vertical
    if h > 260 and w < 120 and aspect < 0.5:
        return 2  # voile

    # Grandes surfaces centrales => dalle pleine ou radier
    if area > 200000:
        if y > h_img * 0.55:
            return 8  # radier (zone inférieure du plan)
        else:
            return 3  # dalle pleine

    # Dalle hourdis : surface moyenne
    if 80000 < area <= 200000:
        return 4

    # Semelle filante : rectangle bas du plan
    if w > 260 and h < 150 and y > h_img * 0.60:
        return 6

    # Semelle isolée : bloc moyen bas du plan
    if 50000 < area < 150000 and y > h_img * 0.55:
        return 5

    # Poutre-voile : rectangle assez haut, pas trop fin
    if h > 250 and 0.5 <= aspect <= 1.0:
        return 7

    return None


# ========== TRAITEMENT PRINCIPAL ==========

def process_image(path):
    img = load_image_any(path)
    if img is None:
        print(f"❌ Impossible de lire {path}")
        return 0

    h_img, w_img = img.shape[:2]

    # On coupe la légende à droite
    cut_w = int(w_img * IGNORE_RIGHT_RATIO)
    work = img[:, :cut_w].copy()

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    bin_img = enhance_image_for_shapes(gray)

    # Contours externes
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area < 1500:
            continue  # trop petit

        aspect = w / h if h > 0 else 0.0

        # Coordonnées dans l'image complète (on recolle la largeur réelle)
        x1, y1, x2, y2 = x, y, x + w, y + h

        # OCR autour de cette zone
        ocr = ocr_text(img, x1, y1, x2, y2)

        cls_id = heuristique_classe_geom_ocr(
            w, h, area, aspect,
            x1, y1, w_img, h_img, ocr
        )
        if cls_id is None:
            continue

        xc, yc, ww, hh = normalize_xyxy(x1, y1, x2, y2, w_img, h_img)
        detections.append((cls_id, xc, yc, ww, hh))

    # Sauvegarde label YOLO
    fname = os.path.basename(path)
    label_path = os.path.join(
        LABEL_DIR,
        fname.replace(".png", ".txt").replace(".jpg", ".txt").replace(".jpeg", ".txt").replace(".pdf", ".txt")
    )

    with open(label_path, "w") as f:
        for cls_id, xc, yc, w, h in detections:
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

    # Sauvegarde image de debug avec rectangles
    debug = img.copy()
    for cls_id, xc, yc, w, h in detections:
        x1 = int((xc - w / 2) * w_img)
        y1 = int((yc - h / 2) * h_img)
        x2 = int((xc + w / 2) * w_img)
        y2 = int((yc + h / 2) * h_img)
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = CLASS_NAMES.get(cls_id, str(cls_id))
        cv2.putText(debug, label, (x1, max(0, y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    debug_path = os.path.join(
        DEBUG_DIR,
        fname.replace(".pdf", ".png")
    )
    cv2.imwrite(debug_path, debug)

    print(f"✓ {fname} → {len(detections)} objets annotés")
    return len(detections)


def main():
    files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".pdf"))
    ]

    if not files:
        print(f"Aucune image/PDF trouvée dans {IMAGE_DIR}")
        return

    print(f"{len(files)} fichiers trouvés. Début de l’annotation…")

    total = 0
    for f in files:
        path = os.path.join(IMAGE_DIR, f)
        total += process_image(path)

    print("===========================================")
    print(f"Annotation terminée. Total d’objets annotés : {total}")
    print(f"Labels YOLO dans : {LABEL_DIR}")
    print(f"Images de debug dans : {DEBUG_DIR}")


if __name__ == "__main__":
    main()
