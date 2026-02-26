from ultralytics import YOLO
from ultralytics.utils.metrics import DetMetrics


def train_yolo(
    data_yaml: str,
    epochs: int = 50,
    model_path: str = "yolo11n.pt",
):
    model = YOLO(model_path)

    results: DetMetrics = model.train( # type: ignore (ultralytics)
        data=data_yaml,
        epochs=epochs,
        imgsz=640,
        batch=8,              # better for small datasets
        name="padel_custom",
        project="experiments/padel_training/logs",
        optimizer="AdamW",
        lr0=0.001,
        pretrained=True,
        exist_ok=True,
    ) 

    print("Training finished!")
    print("Best weights:", results.save_dir + "/weights/best.pt")