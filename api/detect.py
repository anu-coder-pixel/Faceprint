"""
FastAPI endpoint for Faceprint face detection (Vercel serverless compatible).
Provides an interactive web interface on GET / and JSON face detection on POST /api/detect.
Uses OpenCV YuNet model from modules.detection.
"""

import os
import sys
import base64
from io import BytesIO
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image
import numpy as np

# Ensure project root is on sys.path for imports
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from modules.detection import FaceDetector

app = FastAPI(
    title="Faceprint API",
    description="Serverless Face Detection & Verification API for Faceprint",
    version="1.0.0",
)

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


HTML_LANDING_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Faceprint | Serverless Face Detection</title>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --border: #334155;
      --primary: #38bdf8;
      --primary-hover: #0ea5e9;
      --accent: #22c55e;
      --text: #f8fafc;
      --muted: #94a3b8;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
    body { background: var(--bg); color: var(--text); min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 2rem 1rem; }
    .container { width: 100%; max-width: 900px; display: flex; flex-direction: column; gap: 1.5rem; }
    header { text-align: center; margin-bottom: 0.5rem; }
    header h1 { font-size: 2.2rem; font-weight: 800; background: linear-gradient(135deg, #38bdf8, #818cf8); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
    header p { color: var(--muted); margin-top: 0.5rem; font-size: 1rem; }
    .card { background: var(--card); border: 1px solid var(--border); border-radius: 12px; padding: 1.5rem; box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3); }
    .dropzone { border: 2px dashed var(--border); border-radius: 8px; padding: 2.5rem 1rem; text-align: center; cursor: pointer; transition: border-color 0.2s, background-color 0.2s; }
    .dropzone:hover, .dropzone.dragover { border-color: var(--primary); background: rgba(56, 189, 248, 0.05); }
    .dropzone svg { width: 48px; height: 48px; fill: none; stroke: var(--primary); stroke-width: 2; margin-bottom: 0.75rem; }
    .dropzone p { color: var(--muted); font-size: 0.95rem; }
    .dropzone span { color: var(--primary); font-weight: 600; }
    input[type="file"] { display: none; }
    .preview-box { display: none; flex-direction: column; align-items: center; gap: 1rem; margin-top: 1rem; position: relative; }
    canvas { max-width: 100%; height: auto; border-radius: 8px; border: 1px solid var(--border); }
    .btn { background: var(--primary); color: #0f172a; border: none; border-radius: 6px; padding: 0.65rem 1.25rem; font-weight: 600; cursor: pointer; transition: background 0.2s; font-size: 0.95rem; }
    .btn:hover { background: var(--primary-hover); }
    .btn:disabled { opacity: 0.5; cursor: not-allowed; }
    .badge { display: inline-block; padding: 0.25rem 0.6rem; border-radius: 9999px; font-size: 0.8rem; font-weight: 600; }
    .badge-success { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.3); }
    .badge-info { background: rgba(56, 189, 248, 0.2); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.3); }
    .results { margin-top: 1rem; display: flex; flex-direction: column; gap: 0.75rem; }
    .stat-row { display: flex; justify-content: space-between; padding: 0.5rem 0; border-bottom: 1px solid var(--border); font-size: 0.9rem; }
    .stat-label { color: var(--muted); }
    .stat-val { font-weight: 600; }
    pre { background: #0b1120; border: 1px solid var(--border); border-radius: 6px; padding: 1rem; font-size: 0.85rem; overflow-x: auto; color: #cbd5e1; max-height: 250px; }
    footer { text-align: center; color: var(--muted); font-size: 0.85rem; margin-top: 2rem; }
    footer a { color: var(--primary); text-decoration: none; }
    footer a:hover { text-decoration: underline; }
    .loading { display: none; color: var(--primary); font-weight: 600; text-align: center; margin: 1rem 0; }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>👤 Faceprint</h1>
      <p>Serverless Face Detection powered by OpenCV YuNet & Deep Learning</p>
    </header>

    <div class="card">
      <div class="dropzone" id="dropzone" onclick="document.getElementById('file-input').click()">
        <svg viewBox="0 0 24 24"><path d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z"/></svg>
        <p>Drag and drop an image here, or <span>browse file</span></p>
        <p style="font-size:0.8rem; margin-top:0.4rem; color:#64748b;">Supports PNG, JPG, JPEG, WEBP</p>
      </div>
      <input type="file" id="file-input" accept="image/*">

      <div class="loading" id="loading">⚡ Running YuNet deep learning detection...</div>

      <div class="preview-box" id="preview-box">
        <canvas id="canvas"></canvas>
        <div style="display:flex; gap:0.5rem;">
          <button class="btn" onclick="document.getElementById('file-input').click()">Choose Another Image</button>
        </div>
      </div>

      <div class="results" id="results" style="display:none;">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:0.5rem;">
          <h3>Detection Results</h3>
          <span class="badge badge-success" id="face-count-badge">0 Faces Detected</span>
        </div>
        <div id="stats-container"></div>
        <details style="margin-top:0.5rem; cursor:pointer;">
          <summary style="color:var(--primary); font-size:0.9rem; font-weight:600;">View Raw JSON Response</summary>
          <pre id="raw-json" style="margin-top:0.5rem;"></pre>
        </details>
      </div>
    </div>

    <div class="card" style="font-size:0.9rem;">
      <h3 style="margin-bottom:0.5rem; color:#38bdf8;">API Endpoint Usage</h3>
      <p style="color:var(--muted); margin-bottom:0.75rem;">You can call this endpoint programmatically from any frontend, backend, or script:</p>
      <pre>POST /api/detect
Content-Type: application/json

{
  "image_base64": "data:image/jpeg;base64,..."
}</pre>
    </div>

    <footer>
      Faceprint &bull; <a href="https://github.com/anu-coder-pixel/Faceprint" target="_blank">GitHub Repository</a> &bull; Powered by YuNet ONNX
    </footer>
  </div>

  <script>
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const loading = document.getElementById('loading');
    const previewBox = document.getElementById('preview-box');
    const canvas = document.getElementById('canvas');
    const ctx = canvas.getContext('2d');
    const results = document.getElementById('results');
    const faceCountBadge = document.getElementById('face-count-badge');
    const statsContainer = document.getElementById('stats-container');
    const rawJson = document.getElementById('raw-json');

    ['dragenter', 'dragover'].forEach(name => {
      dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.add('dragover'); });
    });
    ['dragleave', 'drop'].forEach(name => {
      dropzone.addEventListener(name, (e) => { e.preventDefault(); dropzone.classList.remove('dragover'); });
    });
    dropzone.addEventListener('drop', (e) => {
      if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', (e) => {
      if (e.target.files.length) handleFile(e.target.files[0]);
    });

    function handleFile(file) {
      if (!file.type.startsWith('image/')) {
        alert('Please select a valid image file.');
        return;
      }
      const reader = new FileReader();
      reader.onload = async (event) => {
        const base64Data = event.target.result;
        const img = new Image();
        img.onload = () => {
          renderCanvas(img, []);
          previewBox.style.display = 'flex';
          dropzone.style.display = 'none';
        };
        img.src = base64Data;
        await detectFaces(base64Data, img);
      };
      reader.readAsDataURL(file);
    }

    async function detectFaces(base64Data, img) {
      loading.style.display = 'block';
      results.style.display = 'none';
      try {
        const response = await fetch('/api/detect', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ image_base64: base64Data })
        });
        const data = await response.json();
        loading.style.display = 'none';
        results.style.display = 'block';
        rawJson.textContent = JSON.stringify(data, null, 2);

        if (data.status === 'success') {
          renderCanvas(img, data.detections || []);
          faceCountBadge.textContent = `${data.faces_detected} Face${data.faces_detected === 1 ? '' : 's'} Detected`;
          renderStats(data.detections || []);
        } else {
          alert('Detection error: ' + (data.detail || 'Unknown error'));
        }
      } catch (err) {
        loading.style.display = 'none';
        alert('Failed to connect to API: ' + err.message);
      }
    }

    function renderCanvas(img, detections) {
      canvas.width = img.naturalWidth || img.width;
      canvas.height = img.naturalHeight || img.height;
      ctx.drawImage(img, 0, 0);

      detections.forEach((det, idx) => {
        const b = det.box;
        ctx.strokeStyle = '#22c55e';
        ctx.lineWidth = Math.max(2, Math.round(canvas.width / 300));
        ctx.strokeRect(b.x, b.y, b.width, b.height);

        // Label badge
        const text = `Face #${idx + 1} (${Math.round(det.confidence * 100)}%)`;
        ctx.font = `bold ${Math.max(14, Math.round(canvas.width / 40))}px sans-serif`;
        const textWidth = ctx.measureText(text).width;
        const textHeight = Math.max(14, Math.round(canvas.width / 40));

        ctx.fillStyle = '#22c55e';
        ctx.fillRect(b.x, Math.max(0, b.y - textHeight - 6), textWidth + 8, textHeight + 6);
        ctx.fillStyle = '#0f172a';
        ctx.fillText(text, b.x + 4, Math.max(textHeight, b.y - 4));

        // Landmarks
        if (det.landmarks) {
          const lColors = { right_eye: '#38bdf8', left_eye: '#ef4444', nose_tip: '#22c55e', right_mouth: '#ec4899', left_mouth: '#eab308' };
          for (const [key, pt] of Object.entries(det.landmarks)) {
            ctx.fillStyle = lColors[key] || '#ffffff';
            ctx.beginPath();
            ctx.arc(pt[0], pt[1], Math.max(3, Math.round(canvas.width / 200)), 0, 2 * Math.PI);
            ctx.fill();
          }
        }
      });
    }

    function renderStats(detections) {
      if (!detections.length) {
        statsContainer.innerHTML = '<p style="color:var(--muted); padding:0.5rem 0;">No faces detected with sufficient confidence in this image.</p>';
        return;
      }
      statsContainer.innerHTML = detections.map((det, i) => `
        <div class="stat-row">
          <span class="stat-label">Face #${i + 1}</span>
          <span class="stat-val" style="color:#22c55e;">${Math.round(det.confidence * 100)}% Confidence &bull; [${det.box.width}x${det.box.height}]</span>
        </div>
      `).join('');
    }
  </script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serves the interactive Faceprint Face Detection web UI."""
    return HTMLResponse(content=HTML_LANDING_PAGE)


@app.get("/health")
@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "Faceprint API", "model": "OpenCV YuNet"}


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
