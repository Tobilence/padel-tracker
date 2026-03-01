from core.detection import Detector, YOLODetector
from core.tracking import PlayerTracker, ByteTrackPlayerTracker
from core.identity import OSNetReIDEmbeddingModel, IdentityManager
from core.viz import vizualize_players
import time
import cv2

class PadelTrackingPipeline:

    def __init__(self, video_path):
        self.detector = YOLODetector(model_path="models/yolo-best-n.pt")
        self.tracker = ByteTrackPlayerTracker()
        self.identity_manager = IdentityManager(OSNetReIDEmbeddingModel())

        self.cap = cv2.VideoCapture(video_path)

    def run(self):

        frame_count = 0
        while True:

            ret, frame = self.cap.read()
            if not ret:
                break

            detections = self.detector.detect(frame)
            tracks = self.tracker.update(detections)
            self.identity_manager.update(frame, tracks)

            frame = vizualize_players(frame, self.identity_manager.players)

            cv2.imshow("Padel Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
            
            # temp: debug slowdown
            # frame_count += 1

            # if frame_count > 500:
            #     time.sleep(5)
