import base64
import os
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
MODEL_PATH = Path(os.getenv("ROADGUARD_MODEL_PATH", ROOT / "models" / "best.pt"))
MAX_IMAGE_BYTES = 10 * 1024 * 1024
ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}

app = FastAPI(title="RoadGuard AI API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False,
                   allow_methods=["GET", "POST"], allow_headers=["*"])
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=503,
                detail="Trained pothole weights are missing. Train a model and place best.pt in backend/models/.")
        try:
            _model = YOLO(str(MODEL_PATH))
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Could not load model: {exc}") from exc
    return _model


def severity_from_box(box, width, height):
    # Visual box-area heuristic only; does not measure pothole depth or real risk.
    x1, y1, x2, y2 = [float(v) for v in box]
    ratio = max(0.0, (x2-x1)*(y2-y1)) / max(1.0, width*height)
    return "High" if ratio >= 0.08 else "Medium" if ratio >= 0.025 else "Low"


@app.get("/")
def root():
    return {"project": "RoadGuard AI", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok", "model_file_found": MODEL_PATH.exists(),
            "model_loaded": _model is not None}


@app.post("/api/detect")
async def detect_potholes(image: UploadFile = File(...), confidence: float = 0.25):
    if image.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=415, detail="Upload a JPG, PNG or WEBP image.")
    if not 0.05 <= confidence <= 0.95:
        raise HTTPException(status_code=400, detail="Confidence must be between 0.05 and 0.95.")
    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="The uploaded image is empty.")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Image is too large. Maximum size is 10 MB.")
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="Could not read this image.")

    result = get_model().predict(source=frame, conf=confidence, verbose=False)[0]
    height, width = frame.shape[:2]
    annotated, detections = frame.copy(), []
    if result.boxes is not None:
        for item in result.boxes:
            coords = item.xyxy[0].cpu().numpy().tolist()
            score = float(item.conf[0].cpu().item())
            class_id = int(item.cls[0].cpu().item())
            label = result.names.get(class_id, str(class_id))
            severity = severity_from_box(coords, width, height)
            x1, y1, x2, y2 = [int(v) for v in coords]
            cv2.rectangle(annotated, (x1,y1), (x2,y2), (40,220,160), 3)
            cv2.putText(annotated, f"{label} {score:.0%} | {severity}",
                (x1,max(24,y1-8)), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                (40,220,160), 2, cv2.LINE_AA)
            detections.append({"label": label, "confidence": round(score,4),
                "severity_estimate": severity, "box_xyxy": [round(v,1) for v in coords]})
    ok, encoded = cv2.imencode(".jpg", annotated)
    if not ok:
        raise HTTPException(status_code=500, detail="Could not encode annotated image.")
    return {"success": True, "image_width": width, "image_height": height,
        "count": len(detections), "detections": detections,
        "severity_note": "Bounding-box-area estimate only; not pothole depth or structural risk.",
        "annotated_image": base64.b64encode(encoded.tobytes()).decode("ascii")}
