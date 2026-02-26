from ultralytics import YOLO

def train_yolo(data_yaml: str, epochs: int = 50, model_path: str = "models/yolov8n.pt"):
    """
    Fine-tune YOLO on custom padel dataset
    """
    # Load base model
    model = YOLO(model_path)
    
    # Train
    model.train(
        data=data_yaml,      # path to data yaml
        epochs=epochs,
        imgsz=640,
        batch=16,
        name="padel_custom",
        project="experiments/padel_training/logs",
        exist_ok=True
    )

    # Save final weights
    model.export(format="pt", imgsz=640)
    print("Fine-tuned weights saved!")