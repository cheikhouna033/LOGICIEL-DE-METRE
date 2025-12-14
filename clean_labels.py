import os

LABEL_DIR = "data/labels/train"

MIN_SIZE = 0.02   # 2%
MAX_SIZE = 0.80   # 80%

for fname in os.listdir(LABEL_DIR):
    if not fname.endswith(".txt"):
        continue

    path = os.path.join(LABEL_DIR, fname)

    with open(path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        cls, x, y, w, h = parts
        w = float(w)
        h = float(h)

        if w < MIN_SIZE or h < MIN_SIZE:
            continue
        if w > MAX_SIZE or h > MAX_SIZE:
            continue

        new_lines.append(line)

    with open(path, "w") as f:
        f.writelines(new_lines)

print("Nettoyage terminé.")
