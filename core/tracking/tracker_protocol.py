from typing import Protocol

class PlayerTracker:

    def update(self, detections):
        ...