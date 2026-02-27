import cv2 


def draw_tracks(frame, tracks):
    """
    Draws bounding boxes and ID labels onto the frame based on tracker results.
    """
    if len(tracks) == 0:
        return frame

    for box, track_id in zip(tracks.xyxy, tracks.tracker_id):
        # Convert coordinates to integers for OpenCV
        x1, y1, x2, y2 = map(int, box)

        # Draw the Bounding Box (Green, thickness 2)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Draw the ID Label slightly above the box
        cv2.putText(
            frame, 
            f"ID {track_id}", 
            (x1, y1 - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.8, 
            (0, 255, 0), 
            2
        )
    
    return frame
