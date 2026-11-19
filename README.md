# 👤 Faceprint: Robust Deep Learning Face Recognition System

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![OpenCV](https://img.shields.io/badge/OpenCV-YuNet%20%7C%20SFace-green.svg)](https://opencv.org/)
[![Gradio](https://img.shields.io/badge/UI-Gradio%206-orange.svg)](https://gradio.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

**Faceprint** is an end-to-end, privacy-focused biometric face recognition and verification system. Powered by lightweight deep learning ONNX models (**OpenCV YuNet** and **SFace**), Faceprint delivers rapid face detection, robust 128-dimensional feature embedding extraction, persistent local vector storage, and flexible 1:1 verification / 1:N identification.

The system is built to remain robust under challenging real-world scenarios including **extreme low-light**, **face masks**, **spectacles/eyewear**, **head tilt & side poses**, and **motion blur**.

---

## 🌟 Key Highlights & Milestones

- **Milestone 1: YuNet Face Detection (`modules/detection.py`)**
  - Ultra-fast ONNX-based deep learning detector.
  - Configurable confidence threshold (default `0.50`), aspect-ratio-preserving down/upscaling, and multi-stage fallback preprocessing (adaptive contrast, CLAHE, and boundary padding) to detect occluded and low-light faces.
- **Milestone 2: SFace Feature Extraction (`modules/feature_extraction.py`)**
  - OpenCV `FaceRecognizerSF` deep neural network extracting compact **128-dimensional L2-normalized embeddings**.
  - Adaptive Gamma correction and LAB-space CLAHE enhancement to compensate for dark and uneven lighting while preserving facial geometry.
- **Milestone 3: Persistent Local Biometric Database (`modules/face_database.py`)**
  - Compressed `.npz` storage with strict validation, unit-norm verification, duplicate name prevention, and multi-photo enrollment per identity.
  - Privacy-preserving architecture: biometric templates never leave your local system.
- **Milestone 4: Matching & Recognition (`modules/matching.py`)**
  - Cosine similarity matching between unit-normalized feature vectors.
  - **1:1 Face Verification** with configurable match thresholds (default `0.363`).
  - **1:N Face Identification** using a hybrid aggregation strategy (`0.70 * max_similarity + 0.30 * centroid_similarity`) to balance peak pose matches with centroid stability across multiple enrolled photos.
- **Milestone 5: Robustness Testing & Benchmarking (`modules/robustness.py`)**
  - Automated benchmark framework evaluating 7 synthetic visual conditions: *original, bright light, low light, side pose, tilted, blurred, noisy, and occluded*.
- **Milestone 6: Modern Interactive Gradio Application (`app.py`)**
  - Dark-mode, responsive web application featuring 5 tabs:
    1. **Register Face**: Enroll new identities with single or multiple reference photos.
    2. **Gallery Manager**: View registered identities, inspect enrolled photos, add additional photos, or delete profiles.
    3. **1:N Face Recognition**: Identify unknown faces from a single image against the entire enrolled gallery.
    4. **1:1 Face Verification**: Verify whether two face images belong to the same individual.
    5. **Ethics & Guidelines**: Best practices, consent notices, and responsible AI considerations.
- **Serverless API Endpoint (`api/detect.py` + `vercel.json`)**
  - FastAPI-powered serverless face detection endpoint for integration into external web services.

---

## 🏗️ System Architecture

```text
[ Input Image ] ──► [ Preprocessing & Scaling ]
                           │
                           ▼
                  [ YuNet Face Detector ]
                  (Bounding Box & 5 Landmarks)
                           │
                           ▼
                  [ Align & Crop (112x112) ]
                           │
                           ▼
            [ Illumination Compensation (CLAHE/Gamma) ]
                           │
                           ▼
                  [ SFace Feature Extractor ]
                  (128-D Unit-L2 Embedding)
                           │
         ┌─────────────────┴─────────────────┐
         ▼                                   ▼
[ Face Database ]                  [ Face Matcher ]
(.npz Vector Store)        (Cosine Similarity & Hybrid Aggregation)
         │                                   │
         └─────────────► 1:N & 1:1 ◄─────────┘
```

---

## 📁 Repository Structure

```text
Faceprint/
├── .gitignore                      # Git exclusion rules (safeguards local databases and secrets)
├── README.md                       # Complete project documentation
├── requirements.txt                # Python package dependencies
├── vercel.json                     # Serverless deployment configuration
├── app.py                          # Full Gradio 6 web application
│
├── api/
│   └── detect.py                   # FastAPI serverless detection endpoint
│
├── models/
│   ├── face_detection_yunet.onnx   # Pretrained YuNet ONNX model (~232 KB)
│   └── face_recognition_sface.onnx # Pretrained SFace ONNX model (~38.7 MB)
│
├── modules/
│   ├── __init__.py                 # Package initializer
│   ├── detection.py                # FaceDetector class (OpenCV YuNet)
│   ├── feature_extraction.py       # FaceFeatureExtractor class (OpenCV SFace)
│   ├── face_database.py            # FaceDatabase class (Local .npz storage)
│   ├── matching.py                 # FaceMatcher class (Cosine matching & 1:N / 1:1)
│   └── robustness.py               # RobustnessEvaluator benchmark suite
│
├── Faceprint_Test_Dataset/         # Synthetic benchmark dataset (7 conditions x 3 identities)
│   ├── faceprint_synthetic_dataset/
│   │   ├── Alex_Rivera/
│   │   ├── Sam_Taylor/
│   │   ├── Taylor_Brooks/
│   │   └── README.txt
│
└── Tests/
    ├── test_detection.py           # Verification of YuNet detection & occlusions
    ├── test_feature_extraction.py  # Verification of SFace 128-D embeddings
    ├── test_face_database.py       # Persistence, multi-photo & integrity tests
    ├── test_matching.py            # 1:1 and 1:N verification tests
    └── test_robustness.py          # Benchmark suite running against test dataset
```

---

## 🚀 Getting Started

### 1. Prerequisites
- Python 3.10, 3.11, or 3.12
- Git

### 2. Installation

Clone the repository and set up a virtual environment:

```bash
# Clone the repository
git clone https://github.com/anu-coder-pixel/Faceprint.git
cd Faceprint

# Create a virtual environment
python -m venv .venv

# Activate the virtual environment
# On Windows (PowerShell):
.venv\Scripts\Activate.ps1
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Launching the Web Application

Start the interactive Gradio interface locally:

```bash
python app.py
```
Open your browser and navigate to `http://127.0.0.1:7860`.

---

## 🧪 Running the Test Suite

Faceprint includes comprehensive automated test suites covering every pipeline stage:

```bash
# 1. Test Face Detection (masks, spectacles, low-light, blur, and scaling)
python test_detection.py

# 2. Test SFace Feature Extraction (dimension, finite values, unit norm)
python test_feature_extraction.py

# 3. Test Face Database (save, load, multi-photo enrollment, duplicate check)
python test_face_database.py

# 4. Test Face Matching & Verification (cosine similarity, 1:1 and 1:N)
python test_matching.py

# 5. Run Full Robustness Benchmark Suite
python test_robustness.py
```

---

## 🔒 Privacy & Biometric Ethics

- **Biometric Data Storage**: All facial embeddings are stored locally on your machine in `database/face_database.npz`. No biometric vectors or facial images are uploaded to external cloud servers.
- **Git Safeguards**: Local `.npz` database files, credentials, and personal photos are explicitly excluded via `.gitignore` to prevent accidental biometric leaks.
- **Responsible Use**: Facial recognition should always be implemented with explicit user consent, transparency, and clear data retention policies.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).
