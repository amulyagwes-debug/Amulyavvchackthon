# RoadGuard AI – AI-Based Pothole Detection & Road Safety Alert System

Hackathon prototype for identifying possible potholes in road images, attaching GPS coordinates and organizing reports for human review.

## Status
- Existing `index.html` frontend is hosted with GitHub Pages.
- FastAPI backend source, YOLO training script and dataset instructions are included.
- **No trained pothole model or dataset is included yet.** Real detections require a properly licensed dataset, training, and `backend/models/best.pt`.
- GitHub Pages hosts the frontend only; it cannot run Python/FastAPI.

## Setup
Follow `backend/README.md` and `backend/dataset/README.md`.

GitHub: https://github.com/amulyagwes-debug/Amulyavvchackthon
Live website: https://amulyagwes-debug.github.io/Amulyavvchackthon/

Safety: severity is a rough bounding-box-area estimate, not pothole depth or actual road risk. Reports need human verification.
