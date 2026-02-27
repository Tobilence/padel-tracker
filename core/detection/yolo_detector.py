from ultralytics import YOLO
import supervision as sv

class YOLODetector:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, frame):
        result = self.model(frame)[0]
        det = sv.Detections.from_ultralytics(result)
        return det