"""
FastAPI endpoint for Faceprint face detection (Vercel serverless compatible).
Uses OpenCV YuNet model from modules.detection.
"""

import os
import sys
import base64
from io import BytesIO
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from PIL import Image
import numpy as np

# Ensure project root is on sys.path for imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules.detection import FaceDetector

app = FastAPI(title="Faceprint API", description="Serverless Face Detection API for Faceprint")

_detector = None


def get_detector() -> FaceDetector:
    """Lazy loader for the FaceDetector model."""
    global _detector
    if _detector is None:
        model_path = os.path.join(ROOT, "models", "face_detection_yunet.onnx")
        _detector = FaceDetector(model_path=model_path, score_threshold=0.5)
    return _detector


class ImagePayload(BaseModel):
    image_base64: str


@app.get("/")
async def root():
    return {"message": "Faceprint API is running", "endpoints": ["/api/detect"]}


@app.post("/api/detect")
async def detect(payload: ImagePayload):
    try:
        raw_b64 = payload.image_base64
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        img_bytes = base64.b64decode(raw_b64)
        pil_img = Image.open(BytesIO(img_bytes)).convert("RGB")
        img_np = np.array(pil_img)
        img_bgr = img_np[:, :, ::-1].copy()
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image data: {exc}")

    try:
        detector = get_detector()
        faces = detector.detect(img_bgr)
        results = []
        if faces is not None and len(faces) > 0:
            for face in faces:
                x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                score = float(face[14])
                landmarks = {
                    "right_eye": [float(face[4]), float(face[5])],
                    "left_eye": [float(face[6]), float(face[7])],
                    "nose_tip": [float(face[8]), float(face[9])],
                    "right_mouth": [float(face[10]), float(face[11])],
                    "left_mouth": [float(face[12]), float(face[13])],
                }
                results.append({
                    "box": {"x": x, "y": y, "width": w, "height": h},
                    "confidence": round(score, 4),
                    "landmarks": landmarks,
                })
        return {
            "status": "success",
            "faces_detected": len(results),
            "detections": results,
        }
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
