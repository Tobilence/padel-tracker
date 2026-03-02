from core.detection import YOLODetector
from core.tracking import ByteTrackPlayerTracker
from core.identity import OSNetReIDEmbeddingModel, IdentityManager
from core.viz import vizualize_players, CourtVizualizer
import cv2

class PadelTrackingPipeline:

    def __init__(self, video_path):
        self.detector = YOLODetector(model_path="models/yolo-best-n.pt")
        self.tracker = ByteTrackPlayerTracker()
        self.identity_manager = IdentityManager(OSNetReIDEmbeddingModel())
        self.court_viz = CourtVizualizer()

        self.cap = cv2.VideoCapture(video_path)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 2000)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                return

            self._handle_frame(frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def sample_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        self._handle_frame(frame)
        return frame
    
    def _handle_frame(self, frame):
        detections = self.detector.detect(frame)
        tracks = self.tracker.update(detections)
        self.identity_manager.update(frame, tracks)

        frame_with_players = vizualize_players(frame.copy(), self.identity_manager.players)
        court_viz = self.court_viz.vizualize_players_on_court(self.identity_manager.player_xyxy)

        cv2.imshow("Padel Tracking", frame_with_players)
        cv2.imshow("Court Live", court_viz)
    

