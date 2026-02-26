from typing import Protocol
import numpy as np
import supervision as sv

class Detector(Protocol):

    def detect(self, frame: np.ndarray) -> sv.Detections:
        ...
