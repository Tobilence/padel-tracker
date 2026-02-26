import cv2
import os
import random

# ---- SETTINGS ----
video_path = "../data/video.mp4"
output_dir = "../data/yolo-finetune-v2"
num_images = 1500

# ------------------

# Create output folder
os.makedirs(output_dir, exist_ok=True)

# Open video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise Exception("Could not open video")

# Get total frame count
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print(f"Total frames in video: {total_frames}")

# Randomly select unique frame indices
frame_indices = sorted(random.sample(range(total_frames), num_images))

saved = 0

for i, frame_idx in enumerate(frame_indices):

    # Jump directly to frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    ret, frame = cap.read()

    if ret:
        filename = os.path.join(output_dir, f"frame_{i:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved += 1

cap.release()

print(f"Saved {saved} images to '{output_dir}'")