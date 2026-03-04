# run_training.py
import os
import sys

# Make sure your module can be imported
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from core.detection.yolo_trainer import train_yolo

if __name__ == "__main__":
    data_yaml = "config.yaml"
    epochs = 50
    model_path = "yolo-best-n.pt"

    train_yolo(data_yaml=data_yaml, epochs=epochs, model_path=model_path)