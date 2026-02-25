import cv2
from ultralytics import YOLO
import supervision as sv

# Load model
model = YOLO("yolo11n.pt")

# Tracker
tracker = sv.ByteTrack()

# Video
cap = cv2.VideoCapture("video.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ---- Detection ----
    result = model(frame)[0]
    detections = sv.Detections.from_ultralytics(result)

    # keep only persons
    detections = detections[detections.class_id == 0]

    # ---- Tracking ----
    tracks = tracker.update_with_detections(detections)

    # ---- Draw results ----
    for box, track_id in zip(tracks.xyxy, tracks.tracker_id):

        x1, y1, x2, y2 = map(int, box)

        # bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2),
                      (0,255,0), 2)

        # ID label
        cv2.putText(
            frame,
            f"ID {track_id}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,255,0),
            2
        )

    # ---- Show live ----
    cv2.imshow("Padel Tracking", frame)

    # press q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()