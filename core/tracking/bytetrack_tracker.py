import supervision as sv

class ByteTrackPlayerTracker:
    """
    Wrapper around Supervision's ByteTrack to provide a unified
    update() interface compatible with PadelTrackingPipeline.
    """

    def __init__(self, max_age: int = 30, n_init: int = 3):
        # Initialize the underlying ByteTrack tracker
        self.tracker = sv.ByteTrack()

    def update(self, detections):
        """
        Update tracker with new detections.

        Args:
            detections (sv.Detections): Detections for the current frame.

        Returns:
            sv.Detections: Detections object with added tracker info:
                           - xyxy (bounding boxes)
                           - tracker_id (unique IDs)
        """
        # Only update if there are detections
        if len(detections) == 0:
            return sv.Detections(xyxy=[], tracker_id=[]) # type: ignore

        # Update tracker
        tracks = self.tracker.update_with_detections(detections)

        # Build a Detections-like object to maintain consistent interface
        tracked_detections = sv.Detections(
            xyxy=tracks.xyxy,
            tracker_id=tracks.tracker_id,
            class_id=tracks.class_id if hasattr(tracks, "class_id") else None,
            confidence=tracks.confidence if hasattr(tracks, "confidence") else None
        )

        return tracked_detections