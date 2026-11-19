"""
Test script for face feature extraction using OpenCV SFace.

This script:
1. Detects the face in 'dataset/test.png' (or 'dataset/test.jpg') using FaceDetector (YuNet).
2. Aligns and crops the detected face to 112x112 using FaceFeatureExtractor.
3. Saves the aligned crop to 'results/aligned_face.png'.
4. Extracts a 128-dimensional embedding from the aligned face.
5. Verifies the embedding shape and checks that all values are finite (no NaN or Inf).
"""

import os
import sys
import cv2
import numpy as np

from modules.detection import FaceDetector
from modules.feature_extraction import FaceFeatureExtractor


def run_test():
    # Define paths
    detection_model_path = os.path.join("models", "face_detection_yunet.onnx")
    recognition_model_path = os.path.join("models", "face_recognition_sface.onnx")

    # Support either test.png or test.jpg
    image_path = os.path.join("dataset", "test.png")
    if not os.path.isfile(image_path):
        image_path = os.path.join("dataset", "test.jpg")

    results_dir = "results"
    aligned_crop_path = os.path.join(results_dir, "aligned_face.png")

    print("=" * 65)
    print(" Faceprint: Face Feature Extraction Test (SFace)")
    print("=" * 65)

    # 1. Error handling: Verify models exist
    if not os.path.isfile(detection_model_path):
        print(f"\n[ERROR] Face detection model not found at: '{detection_model_path}'")
        print("Please ensure 'face_detection_yunet.onnx' is placed in the 'models/' folder.")
        sys.exit(1)

    if not os.path.isfile(recognition_model_path):
        print(f"\n[ERROR] Face recognition model not found at: '{recognition_model_path}'")
        print("Please ensure 'face_recognition_sface.onnx' is placed in the 'models/' folder.")
        sys.exit(1)

    # 2. Error handling: Verify test image exists
    if not os.path.isfile(image_path):
        print(f"\n[ERROR] Test image not found at '{image_path}'.")
        print("Please place a test image named 'test.png' or 'test.jpg' inside the 'dataset/' folder.")
        sys.exit(1)

    os.makedirs(results_dir, exist_ok=True)

    # 3. Load image
    print(f"\n[1/5] Loading test image from: '{image_path}'...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Failed to decode image from '{image_path}'. File may be corrupt.")
        sys.exit(1)

    h, w, c = image.shape
    print(f"      Image loaded successfully! Dimensions: {w}x{h} ({c} channels)")

    # 4. Detect face using YuNet
    print(f"\n[2/5] Detecting faces using YuNet...")
    detector = FaceDetector(model_path=detection_model_path, score_threshold=0.8)
    faces = detector.detect_faces(image)

    if faces is None or len(faces) == 0:
        print("[ERROR] No faces detected in the image. Cannot extract features without a detected face.")
        sys.exit(1)

    num_faces = len(faces)
    face = faces[0]  # Take the primary detected face
    confidence = float(face[14])
    print(f"      Detected {num_faces} face(s). Using primary face with confidence: {confidence * 100:.2f}%")

    # 5. Initialize SFace feature extractor
    print(f"\n[3/5] Initializing FaceFeatureExtractor (SFace)...")
    try:
        extractor = FaceFeatureExtractor(model_path=recognition_model_path)
        print("      SFace feature extractor initialized successfully!")
    except Exception as e:
        print(f"[ERROR] Failed to initialize FaceFeatureExtractor: {e}")
        sys.exit(1)

    # 6. Align and crop detected face
    print(f"\n[4/5] Aligning and cropping face using facial landmarks...")
    try:
        aligned_face = extractor.align_crop(image, face)
        print(f"      Face aligned and cropped! Aligned image shape: {aligned_face.shape}")

        # Save aligned crop for visual inspection
        cv2.imwrite(aligned_crop_path, aligned_face)
        print(f"      Saved aligned face crop to: '{aligned_crop_path}'")
    except Exception as e:
        print(f"[ERROR] Face alignment failed: {e}")
        sys.exit(1)

    # 7. Extract feature embedding
    print(f"\n[5/5] Extracting 128-D feature embedding...")
    try:
        embedding = extractor.extract_feature(aligned_face)
    except Exception as e:
        print(f"[ERROR] Feature extraction failed: {e}")
        sys.exit(1)

    # 8. Verify embedding shape and finite numerical values
    print("\n" + "-" * 65)
    print(" Embedding Verification Results:")
    print("-" * 65)

    print(f"  - NumPy Array Shape : {embedding.shape}")
    print(f"  - Data Type         : {embedding.dtype}")
    print(f"  - Total Dimensions  : {embedding.size}")

    # Check for NaN / Inf
    has_nan = np.isnan(embedding).any()
    has_inf = np.isinf(embedding).any()
    is_finite = np.isfinite(embedding).all()

    print(f"  - Contains NaN?     : {has_nan}")
    print(f"  - Contains Inf?     : {has_inf}")
    print(f"  - All values finite?: {is_finite}")

    if not is_finite:
        print("\n[FAIL] Embedding contains invalid numerical values!")
        sys.exit(1)

    l2_norm = float(np.linalg.norm(embedding))
    print(f"  - L2 Norm (Length)  : {l2_norm:.4f}")
    print(f"  - Value Range       : [{embedding.min():.4f}, {embedding.max():.4f}]")

    # Display preview of first 8 dimensions
    preview_dims = embedding.flatten()[:8]
    preview_str = ", ".join(f"{v:.4f}" for v in preview_dims)
    print(f"  - Preview (first 8) : [{preview_str}, ...]")

    print("\n" + "=" * 65)
    print(" [SUCCESS] Face feature extraction test completed successfully!")
    print("=" * 65)


if __name__ == "__main__":
    run_test()

