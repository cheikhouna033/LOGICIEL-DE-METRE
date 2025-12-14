import os
import cv2
import numpy as np

# ========= CONFIG =========
IMAGE_DIR = "data/images/train"
LABEL_DIR = "data/labels/train"
os.makedirs(LABEL_DIR, exist_ok=True)

# Si tu veux ignorer la légende à droite du plan (zone texte),
# on ne garde que la partie gauche (80 % de la largeur).
IGNORE_RIGHT_RATIO = 0.80  # x > 0.80 * largeur => ignoré


def normalize_xyxy(x1, y1, x2, y2, w_img, h_img):
    """Convertit un rectangle (pixels) en format YOLO normalisé."""
    xc = (x1 + x2) / 2 / w_img
    yc = (y1 + y2) / 2 / h_img
    w = (x2 - x1) / w_img
    h = (y2 - y1) / h_img
    return xc, yc, w, h


def classify_element(w, h, area, aspect, x, y, w_img, h_img):
    """
    Classe heuristique basée sur la géométrie.
    Retourne l'ID de classe YOLO (0 à 8).
    """

    # Poteau : petit rectangle quasi carré
    if 40 < w < 200 and 40 < h < 200 and 0.8 < aspect < 1.25:
        return 1  # poteau

    # Poutre horizontale : rectangle long et mince
    if w > 200 and h < 120 and aspect > 3.0:
        return 0  # poutre

    # Voile vertical : rectangle haut et mince
    if h > 250 and w < 120 and aspect < 0.5:
        return 2  # voile

    # Grandes surfaces centrales => dalle pleine ou radier
    if area > 200000:
        # Si zone très basse dans le plan => radier probable
        if y > h_img * 0.55:
            return 8  # radier
        else:
            return 3  # dalle pleine

    # Dalle hourdis : surface moyenne, souvent rectangulaire
    if 80000 < area <= 200000:
        return 4  # dalle hourdis

    # Semelle filante : long rectangle proche du bas du plan
    if w > 250 and h < 150 and y > h_img * 0.60:
        return 6  # semelle filante

    # Semelle isolée : rectangle assez massif dans la zone inférieure
    if 50000 < area < 150000 and y > h_img * 0.55:
        return 5  # semelle isolée

    # Poutre-voile : rectangle haut mais pas trop fin
    if h > 250 and 0.5 <= aspect <= 1.0:
        return 7  # poutre-voile

    # Par défaut, on ne garde pas (None => pas annoté)
    return None


def process_image(img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Impossible de lire {img_path}")
        return []

    h_img, w_img = img.shape[:2]

    # On recadre pour éviter la légende à droite
    work_width = int(w_img * IGNORE_RIGHT_RATIO)
    work_img = img[:, :work_width].copy()

    gray = cv2.cvtColor(work_img, cv2.COLOR_BGR2GRAY)
    # Renforcer les lignes
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)

    # Dilatation pour fusionner un peu les traits
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)

    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    detections = []

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        aspect = w / h if h != 0 else 0

        # Filtrer le bruit (très petits objets)
        if w < 30 or h < 30 or area < 1000:
            continue

        cls_id = classify_element(w, h, area, aspect, x, y, work_width, h_img)
        if cls_id is None:
            continue

        x1, y1, x2, y2 = x, y, x + w, y + h
        xc, yc, ww, hh = normalize_xyxy(x1, y1, x2, y2, w_img, h_img)
        detections.append((cls_id, xc, yc, ww, hh))

    return detections


def main():
    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    if not image_files:
        print(f"Aucune image trouvée dans {IMAGE_DIR}")
        return

    print(f"{len(image_files)} images trouvées. Génération des annotations…")

    for fname in image_files:
        img_path = os.path.join(IMAGE_DIR, fname)
        detections = process_image(img_path)

        label_path = os.path.join(
            LABEL_DIR,
            fname.replace(".png", ".txt").replace(".jpg", ".txt").replace(".jpeg", ".txt")
        )

        with open(label_path, "w") as f:
            for cls_id, xc, yc, w, h in detections:
                f.write(f"{cls_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n")

        print(f"✓ {fname} → {len(detections)} objets annotés")

    print("🎉 Annotation semi-automatique terminée !")
    print("➡ Vérifie/rectifie maintenant avec LabelImg.")


if __name__ == "__main__":
    main()
