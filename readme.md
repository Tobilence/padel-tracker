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

## Status

This project is still a work in progress. Results are promising, and the next steps are improving robustness, reducing ID switches, and expanding evaluation across more videos.
