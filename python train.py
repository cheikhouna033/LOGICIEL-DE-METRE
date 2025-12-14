from ultralytics import YOLO

def main():
    model = YOLO("yolov8n.pt")  # modèle léger (CPU OK)

    model.train(
        data="data.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        name="metre_final"
    )

if __name__ == "__main__":
    main()
