from pdf2image import convert_from_path
import os

PDF_PATH = "COFFRAGES_TRAIN MODEL.pdf"
OUTPUT_DIR = "data/images/train"  # images brutes

os.makedirs(OUTPUT_DIR, exist_ok=True)

pages = convert_from_path(
    PDF_PATH,
    poppler_path=r"C:\poppler-25.12.0\Library\bin"
)

for i, page in enumerate(pages):
    file_path = os.path.join(OUTPUT_DIR, f"plan_{i+1:03d}.png")
    page.save(file_path, "PNG")
    print("Image créée :", file_path)

print("🎉 Conversion PDF → Images terminée !")
