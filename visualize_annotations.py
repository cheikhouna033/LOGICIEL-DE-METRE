import cv2
import os

DIR = "data/images/train/"
LABEL_DIR = "data/labels/train/"

NAMES = {
    0: "poutre",
    1: "poteau",
    2: "voile",
    3: "dalle_pleine",
    4: "semelle_isolee",
    5: "semelle_filante",
    6: "poutre_voile",
    7: "radier",
    8: "inconnu"
}

def show(img_path):
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    txt = img_path.replace("images", "labels").replace(".png", ".txt")
    if not os.path.exists(txt):
        print("Pas de label pour :", img_path)
        return

    with open(txt, "r") as f:
        lines = f.readlines()

    for line in lines:
        cls, xc, yc, ww, hh = map(float, line.split())
        x1 = int((xc - ww/2) * w)
        y1 = int((yc - hh/2) * h)
        x2 = int((xc + ww/2) * w)
        y2 = int((yc + hh/2) * h)
        cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
        cv2.putText(img, NAMES[int(cls)], (x1,y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

    cv2.imshow("Visualization", img)
    cv2.waitKey(0)

for file in os.listdir(DIR):
    if file.endswith(".png"):
        show(DIR + file)
