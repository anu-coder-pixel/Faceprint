"""
Test script for face detection using OpenCV YuNet.

This script:
1. Tests the FaceDetector module on 'dataset/test.png' (or 'dataset/test.jpg')
   and saves the annotated output to 'results/detection_result.jpg'.
2. Runs additional verification tests on simulated challenging conditions:
   - Normal face
   - Face with spectacles
   - Face with a mask (lower-face occlusion)
   - Low-light face
   - Blurred face
"""

import os
import sys
import cv2
import numpy as np
from modules.detection import FaceDetector


def run_condition_tests(detector: FaceDetector, base_image: np.ndarray, primary_face: np.ndarray):
    """Tests detection across normal, spectacles, masked, low-light, and blurred variations."""
    print("\n" + "=" * 60)
    print(" Additional Challenging Condition Detection Tests")
    print("=" * 60)

    x, y, w, h = primary_face[0:4].astype(int)

    # 1. Normal face
    normal_img = base_image.copy()

    # 2. Face with spectacles (simulated dark rectangular glasses frames over eye region)
    spectacles_img = base_image.copy()
    cv2.rectangle(
        spectacles_img,
        (x + int(w * 0.14), y + int(h * 0.30)),
        (x + int(w * 0.45), y + int(h * 0.45)),
        (25, 25, 25),
        thickness=4,
    )
    cv2.rectangle(
        spectacles_img,
        (x + int(w * 0.55), y + int(h * 0.30)),
        (x + int(w * 0.86), y + int(h * 0.45)),
        (25, 25, 25),
        thickness=4,
    )
    cv2.line(
        spectacles_img,
        (x + int(w * 0.45), y + int(h * 0.37)),
        (x + int(w * 0.55), y + int(h * 0.37)),
        (25, 25, 25),
        thickness=4,
    )

    # 3. Face with a mask (covering nose bridge, mouth, and chin)
    masked_img = base_image.copy()
    cv2.rectangle(
        masked_img,
        (x - 8, int(y + h * 0.48)),
        (x + w + 8, y + h + 15),
        (210, 220, 230),
        thickness=-1,
    )

    # 4. Low-light face (gamma / brightness darkened by ~70%)
    low_light_img = np.clip(base_image.astype(np.float32) * 0.30, 0, 255).astype(np.uint8)

    # 5. Blurred face (Gaussian blur kernel 15x15)
    blurred_img = cv2.GaussianBlur(base_image, (15, 15), sigmaX=4.0)

    # 5. High-resolution (aspect-preserving downscale) and 4-channel RGBA check
    high_res_img = cv2.resize(base_image, (2800, 2244))
    rgba_img = cv2.cvtColor(base_image, cv2.COLOR_BGR2BGRA)
    tight_crop_img = base_image[max(0, y):min(base_image.shape[0], y + h), max(0, x):min(base_image.shape[1], x + w)]

    test_cases = [
        ("Normal face", normal_img),
        ("Face with spectacles", spectacles_img),
        ("Face with a mask", masked_img),
        ("Low-light face", low_light_img),
        ("Blurred face", blurred_img),
        ("High-res (2800x2244)", high_res_img),
        ("RGBA 4-channel PNG", rgba_img),
        ("Tight crop edge test", tight_crop_img),
    ]

    all_passed = True
    for label, test_img in test_cases:
        detected = detector.detect_faces(test_img)
        count = 0 if detected is None else len(detected)
        if count > 0:
            top_conf = float(detected[0][14]) * 100.0
            print(f"  [PASS] {label:<24} -> Detected {count} face(s) (top confidence: {top_conf:.1f}%)")
        else:
            print(f"  [FAIL] {label:<24} -> No face detected above threshold ({detector.score_threshold:.2f})")
            all_passed = False

    # Verify threshold configurability (a threshold of 0.99 should filter out ~0.92 detection)
    strict_faces = detector.detect_faces(normal_img, score_threshold=0.99)
    assert strict_faces is None or len(strict_faces) == 0, "Per-call strict threshold 0.99 did not filter detection!"
    assert detector.score_threshold == 0.5, "Original detector threshold was not restored after per-call override!"
    print("  [PASS] Threshold config       -> Default 0.50 verified & per-call override 0.99 filtered cleanly")

    if not all_passed:
        print("\n[ERROR] One or more challenging conditions failed detection.")
        sys.exit(1)
    else:
        print("\n[SUCCESS] All condition & preprocessing tests detected cleanly!")


def run_test():
    # Define paths
    model_path = os.path.join("models", "face_detection_yunet.onnx")
    image_path = os.path.join("dataset", "test.png")
    if not os.path.isfile(image_path):
        image_path = os.path.join("dataset", "test.jpg")
    results_dir = "results"
    output_path = os.path.join(results_dir, "detection_result.jpg")

    print("=" * 60)
    print(" Faceprint: Face Detection Test (YuNet)")
    print("=" * 60)

    # 1. Error handling: Check if model file exists
    if not os.path.isfile(model_path):
        print(f"\n[ERROR] Model file not found at: '{model_path}'")
        print("Please ensure 'face_detection_yunet.onnx' is placed in the 'models/' folder.")
        sys.exit(1)

    # 2. Error handling: Check if test image exists
    if not os.path.isfile(image_path):
        print(f"\n[ERROR] Test image not found at: '{image_path}'")
        print("Please place a test image named 'test.png' or 'test.jpg' inside the 'dataset/' folder.")
        sys.exit(1)

    # 3. Ensure results directory exists
    os.makedirs(results_dir, exist_ok=True)

    # 4. Load the input image
    print(f"\n[1/4] Loading image from: '{image_path}'...")
    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Failed to decode image from '{image_path}'. File may be corrupt or not a supported image format.")
        sys.exit(1)

    h, w, c = image.shape
    print(f"      Image loaded successfully! Dimensions: {w}x{h} ({c} channels)")

    # 5. Initialize FaceDetector (verifies default score_threshold == 0.50)
    print(f"\n[2/4] Initializing FaceDetector with model: '{model_path}'...")
    try:
        detector = FaceDetector(model_path=model_path)
        assert detector.score_threshold == 0.50, f"Expected default 0.50, got {detector.score_threshold}"
        print(f"      YuNet face detector initialized successfully! (default score_threshold={detector.score_threshold:.2f})")
    except Exception as e:
        print(f"[ERROR] Failed to initialize FaceDetector: {e}")
        sys.exit(1)

    # 6. Detect faces
    print(f"\n[3/4] Running face detection...")
    try:
        faces = detector.detect_faces(image)
    except Exception as e:
        print(f"[ERROR] Exception during face detection: {e}")
        sys.exit(1)

    num_faces = 0 if faces is None else len(faces)
    print(f"      Detection completed: {num_faces} face(s) found.")

    if faces is not None and len(faces) > 0:
        for idx, face in enumerate(faces, start=1):
            x, y, fw, fh = face[0:4].astype(int)
            confidence = float(face[14])
            print(f"      - Face #{idx}: Bounding Box=(x={x}, y={y}, w={fw}, h={fh}), Confidence={confidence * 100:.2f}%")
    else:
        print(f"      [NOTE] No faces detected above the confidence threshold ({detector.score_threshold:.2f}).")

    # 7. Draw bounding boxes and save output
    print(f"\n[4/4] Annotating image and saving result to: '{output_path}'...")
    annotated_image = detector.draw_faces(image, faces)

    success = cv2.imwrite(output_path, annotated_image)
    if not success:
        print(f"[ERROR] Failed to write annotated image to '{output_path}'.")
        sys.exit(1)

    print(f"      Annotated image successfully saved to '{output_path}'!")

    # 8. Run additional condition verification tests if a baseline face was found
    if faces is not None and len(faces) > 0:
        run_condition_tests(detector, image, faces[0])

    print("\n" + "=" * 60)
    print(" Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    run_test()
