# Padel Video Tracking

![Padel tracking demo](assets/sample.gif)

Work in progress padel analytics pipeline for tracking players across match footage.

## What this project does

- Detects padel players in video frames
- Tracks identities across time
- Produces a basis for later analytics and insights

## Current approach

- **Detection:** YOLO fine-tuned for padel-player detection
- **Re-identification:** OSNet-based player re-ID to keep identities consistent
- **Pipeline goal:** robust player tracking in real match scenarios
