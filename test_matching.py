"""
Test script for Milestone 4: Face Matching and Recognition.

This script tests:
1. Matching an embedding against itself (Self-Matching identity property: similarity ≈ 1.0).
2. Matching two embeddings from the same person (1:1 Verification Match).
3. Matching embeddings from different people (1:1 Verification Mismatch).
4. Empty database handling during 1:N identification (graceful 'Unknown' handling).
5. 1:N Identification of a registered identity against multiple stored gallery photos.
6. Unknown-person rejection (1:N probe below threshold correctly classified as 'Unknown').
7. End-to-end 1:1 verification directly from raw image matrices.
"""

import os
import sys
import cv2
import numpy as np

from modules.detection import FaceDetector
from modules.feature_extraction import FaceFeatureExtractor
from modules.face_database import FaceDatabase
from modules.matching import FaceMatcher


def run_tests():
    print("=" * 70)
    print(" Faceprint: Milestone 4 - Face Matching & Recognition Test")
    print("=" * 70)

    # 1. Verify environment and prerequisites
    detection_model_path = os.path.join("models", "face_detection_yunet.onnx")
    recognition_model_path = os.path.join("models", "face_recognition_sface.onnx")

    image_path = os.path.join("dataset", "test.png")
    if not os.path.isfile(image_path):
        image_path = os.path.join("dataset", "test.jpg")

    if not os.path.isfile(detection_model_path) or not os.path.isfile(recognition_model_path):
        print("[ERROR] Required ONNX models not found in 'models/' directory.")
        sys.exit(1)

    if not os.path.isfile(image_path):
        print(f"[ERROR] Test image not found at '{image_path}'.")
        sys.exit(1)

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Failed to load image from '{image_path}'.")
        sys.exit(1)

    # Extract real 128-D face embedding from dataset photo
    print("\n[Setup] Extracting real face embedding from dataset photo...")
    detector = FaceDetector(model_path=detection_model_path)
    extractor = FaceFeatureExtractor(model_path=recognition_model_path)

    faces = detector.detect_faces(image)
    if faces is None or len(faces) == 0:
        print("[ERROR] No face detected in test image.")
        sys.exit(1)

    aligned_crop = extractor.align_crop(image, faces[0])
    person1_emb1 = extractor.extract_feature(aligned_crop)
    assert abs(float(np.linalg.norm(person1_emb1)) - 1.0) < 1e-4, "Embedding must be unit-L2 normalized!"
    print(f"  ✓ Real embedding extracted successfully (shape: {person1_emb1.shape}, L2 norm: {np.linalg.norm(person1_emb1):.4f})")

    # Generate synthetic variations and different identities for testing:
    # - person1_emb2: Same person under simulated variation (subtle Gaussian perturbation)
    rng = np.random.RandomState(42)
    perturbation = rng.normal(0, 0.015, size=person1_emb1.shape).astype(np.float32)
    person1_emb2 = person1_emb1 + perturbation

    # - person2_emb: Completely different person (orthogonal random vector)
    person2_emb = rng.normal(0, 1.0, size=(1, 128)).astype(np.float32)
    person2_emb /= np.linalg.norm(person2_emb)

    # - unknown_stranger_emb: A third distinct individual not in gallery
    unknown_stranger_emb = rng.normal(0, 1.0, size=(1, 128)).astype(np.float32)
    unknown_stranger_emb /= np.linalg.norm(unknown_stranger_emb)

    matcher = FaceMatcher(detector=detector, extractor=extractor, threshold=0.363)

    # -------------------------------------------------------------
    # TEST 1: Matching an embedding against itself (Self-Match)
    # -------------------------------------------------------------
    print("\n[Test 1/7] Matching an embedding against itself (Self-Match)...")
    self_sim = matcher.compute_cosine_similarity(person1_emb1, person1_emb1)
    print(f"  - Self-similarity score: {self_sim:.6f} (Theoretical target: 1.000000)")
    assert abs(self_sim - 1.0) < 1e-4, f"Self-similarity should be ~1.0, got {self_sim}"

    self_verify = matcher.verify(person1_emb1, person1_emb1)
    assert self_verify["verified"] is True, "Self-verification failed!"
    print(f"  ✓ Self-verification passed: verified={self_verify['verified']}, status='{self_verify['status']}'")

    # -------------------------------------------------------------
    # TEST 2: Matching two embeddings from the same person
    # -------------------------------------------------------------
    print("\n[Test 2/7] Matching two embeddings from the SAME person...")
    same_person_verify = matcher.verify(person1_emb1, person1_emb2)
    print(f"  - Cosine similarity: {same_person_verify['similarity']:.4f} (Threshold: {same_person_verify['threshold']})")
    assert same_person_verify["verified"] is True, "Same-person verification should match!"
    print(f"  ✓ Same-person test passed: verified={same_person_verify['verified']}, status='{same_person_verify['status']}'")

    # -------------------------------------------------------------
    # TEST 3: Matching embeddings from DIFFERENT people
    # -------------------------------------------------------------
    print("\n[Test 3/7] Matching embeddings from DIFFERENT people...")
    diff_person_verify = matcher.verify(person1_emb1, person2_emb)
    print(f"  - Cosine similarity: {diff_person_verify['similarity']:.4f} (Threshold: {diff_person_verify['threshold']})")
    assert diff_person_verify["verified"] is False, "Different-person verification should NOT match!"
    print(f"  ✓ Different-person test passed: verified={diff_person_verify['verified']}, status='{diff_person_verify['status']}'")

    # -------------------------------------------------------------
    # TEST 4: Empty database handling (1:N identification)
    # -------------------------------------------------------------
    print("\n[Test 4/7] Handling 1:N identification on an EMPTY database...")
    empty_db = FaceDatabase(auto_load=False)
    matcher_empty_db = FaceMatcher(database=empty_db, threshold=0.363)
    empty_result = matcher_empty_db.identify(person1_emb1)
    print(f"  - Result Identity : '{empty_result['identity']}'")
    print(f"  - Result Status   : '{empty_result['status']}'")
    assert empty_result["identity"] == "Unknown", "Empty database must return 'Unknown'!"
    print("  ✓ Empty database handled gracefully without crashing")

    # -------------------------------------------------------------
    # TEST 5: 1:N Face Identification with Enrolled Gallery
    # -------------------------------------------------------------
    print("\n[Test 5/7] 1:N Identification against enrolled gallery...")
    db = FaceDatabase(auto_load=False)
    # Enroll Person 1 with multiple photos
    db.register_face("Alex Rivera", person1_emb1)
    db.add_embedding("Alex Rivera", person1_emb2)
    # Enroll Person 2
    db.register_face("Taylor Brooks", person2_emb)

    matcher_with_db = FaceMatcher(database=db, threshold=0.363, aggregation="max")

    # Probe with Person 1's second photo
    id_result = matcher_with_db.identify(person1_emb2)
    print(f"  - Matched Identity: '{id_result['identity']}'")
    print(f"  - Similarity Score: {id_result['similarity']:.4f}")
    print(f"  - Status          : '{id_result['status']}'")
    print(f"  - Candidates      : {id_result['candidates']}")

    assert id_result["identity"] == "Alex Rivera", f"Expected 'Alex Rivera', got {id_result['identity']}"
    assert id_result["status"] == "Recognized", f"Expected 'Recognized', got {id_result['status']}"
    print("  ✓ 1:N identification correctly identified 'Alex Rivera'")

    # -------------------------------------------------------------
    # TEST 6: Unknown-person rejection (1:N probe below threshold)
    # -------------------------------------------------------------
    print("\n[Test 6/7] Unknown-person rejection (Probe not in gallery)...")
    unknown_result = matcher_with_db.identify(unknown_stranger_emb)
    print(f"  - Matched Identity: '{unknown_result['identity']}'")
    print(f"  - Top Similarity  : {unknown_result['similarity']:.4f} (Threshold: {unknown_result['threshold']})")
    print(f"  - Status          : '{unknown_result['status']}'")

    assert unknown_result["identity"] == "Unknown", f"Expected 'Unknown', got {unknown_result['identity']}"
    assert "Below threshold" in unknown_result["status"], f"Expected status 'Below threshold', got {unknown_result['status']}"
    print("  ✓ Unknown person successfully rejected as 'Unknown'")

    # -------------------------------------------------------------
    # TEST 7: End-to-end 1:1 image verification from raw BGR images
    # -------------------------------------------------------------
    print("\n[Test 7/9] End-to-end 1:1 verification directly from raw image...")
    img_verify_result = matcher.verify(image, image)
    print(f"  - Raw Image Verification: verified={img_verify_result['verified']}, similarity={img_verify_result['similarity']:.4f}")
    assert img_verify_result["verified"] is True, "Image-to-image self-verification failed!"
    print("  ✓ End-to-end image verification passed successfully")

    # -------------------------------------------------------------
    # TEST 8: Multi-Condition & Low-Light 1:N Recognition (Threshold = 0.363)
    # -------------------------------------------------------------
    print("\n[Test 8/9] 1:N Recognition under normal, low-light, spectacles, mask, and blur...")
    fx, fy, fw, fh = faces[0][0:4].astype(int)

    # Create controlled variations from dataset photo (preserving original image)
    cond_normal = image.copy()
    cond_very_dark = np.clip(image.astype(np.float32) * 0.04, 0, 255).astype(np.uint8)
    cond_dark_noisy = np.clip(
        image.astype(np.float32) * 0.05 + rng.normal(0, 2.5, image.shape), 0, 255
    ).astype(np.uint8)

    cond_spectacles = image.copy()
    cv2.rectangle(cond_spectacles, (fx + int(fw * 0.14), fy + int(fh * 0.30)), (fx + int(fw * 0.45), fy + int(fh * 0.45)), (25, 25, 25), 4)
    cv2.rectangle(cond_spectacles, (fx + int(fw * 0.55), fy + int(fh * 0.30)), (fx + int(fw * 0.86), fy + int(fh * 0.45)), (25, 25, 25), 4)
    cv2.line(cond_spectacles, (fx + int(fw * 0.45), fy + int(fh * 0.37)), (fx + int(fw * 0.55), fy + int(fh * 0.37)), (25, 25, 25), 4)

    cond_masked = image.copy()
    cv2.rectangle(cond_masked, (fx - 8, int(fy + fh * 0.48)), (fx + fw + 8, fy + fh + 15), (210, 220, 230), -1)

    cond_blurred = cv2.GaussianBlur(image, (15, 15), sigmaX=4.0)

    # Multi-photo gallery enrollment (normal + slight dim + spectacles) using identical preprocessing
    multi_db = FaceDatabase(auto_load=False)
    multi_db.register_face("Subject_Enrolled", extractor.extract_from_face(cond_normal, faces[0]))
    cond_dim = np.clip(image.astype(np.float32) * 0.25, 0, 255).astype(np.uint8)
    dim_faces = detector.detect_faces(cond_dim)
    if dim_faces is not None and len(dim_faces) > 0:
        multi_db.add_embedding("Subject_Enrolled", extractor.extract_from_face(cond_dim, dim_faces[0]))

    hybrid_matcher = FaceMatcher(
        database=multi_db,
        detector=detector,
        extractor=extractor,
        threshold=0.363,
        aggregation="hybrid",
    )

    condition_probes = [
        ("Normal", cond_normal),
        ("Very Dark (0.04x)", cond_very_dark),
        ("Dark + Noise (0.05x)", cond_dark_noisy),
        ("Spectacles", cond_spectacles),
        ("Masked", cond_masked),
        ("Blurred", cond_blurred),
    ]

    for cond_name, probe_img in condition_probes:
        res = hybrid_matcher.identify(probe_img, threshold=0.363)
        print(
            f"  - {cond_name:<22} -> Identity='{res['identity']}', Similarity={res['similarity']:.4f}, Status='{res['status']}'"
        )
        assert res["identity"] == "Subject_Enrolled", (
            f"Condition '{cond_name}' failed recognition at standard 0.363 threshold (sim={res['similarity']:.4f})"
        )

    # Verify raw baseline WITHOUT illumination preprocessing fails or struggles on 0.04x dark
    raw_extractor = FaceFeatureExtractor(model_path=recognition_model_path, normalize_illumination=False)
    raw_sim_dark = FaceMatcher.compute_cosine_similarity(
        raw_extractor.extract_from_face(cond_normal, faces[0]),
        raw_extractor.extract_from_face(cond_very_dark, detector.detect_faces(cond_very_dark)[0]),
    )
    pre_sim_dark = FaceMatcher.compute_cosine_similarity(
        extractor.extract_from_face(cond_normal, faces[0]),
        extractor.extract_from_face(cond_very_dark, detector.detect_faces(cond_very_dark)[0]),
    )
    print(f"  ✓ Very Dark (0.04x) similarity boosted from {raw_sim_dark:.4f} (raw) -> {pre_sim_dark:.4f} (CLAHE+Gamma)")
    assert pre_sim_dark > raw_sim_dark, "Controlled preprocessing should improve very dark similarity!"

    # -------------------------------------------------------------
    # TEST 9: Impostor Rejection (Different person NOT recognized as enrolled)
    # -------------------------------------------------------------
    print("\n[Test 9/9] Verifying distinct person rejection & no false match...")
    impostor_path = os.path.join("Faceprint_Test_Dataset", "faceprint_synthetic_dataset", "Taylor_Brooks", "original.png")
    if os.path.isfile(impostor_path):
        impostor_img = cv2.imread(impostor_path)
        impostor_res = hybrid_matcher.identify(impostor_img, threshold=0.363)
        print(
            f"  - Distinct person (Taylor_Brooks vs Subject_Enrolled): Identity='{impostor_res['identity']}', Similarity={impostor_res['similarity']:.4f}"
        )
        assert impostor_res["identity"] == "Unknown", (
            f"False acceptance! Distinct person matched as {impostor_res['identity']} (sim={impostor_res['similarity']:.4f})"
        )
        print("  ✓ Distinct unenrolled person cleanly rejected as 'Unknown'")

    # Read-only check against user's existing database (if present) ensuring no false match as 'Alex'
    user_db_path = os.path.join("database", "face_database.npz")
    if os.path.isfile(user_db_path):
        user_db = FaceDatabase(db_path=user_db_path, auto_load=True)
        if "Alex" in user_db and "Phanishree" in user_db:
            user_matcher = FaceMatcher(database=user_db, detector=detector, extractor=extractor, threshold=0.363, aggregation="hybrid")
            phan_emb = user_db.get_representative_embedding("Phanishree")
            # Evaluate Phanishree against Alex's templates only
            alex_only_db = FaceDatabase(auto_load=False)
            alex_only_db.register_face("Alex", user_db.get_embeddings("Alex"))
            alex_matcher = FaceMatcher(database=alex_only_db, detector=detector, extractor=extractor, threshold=0.363, aggregation="hybrid")
            cross_res = alex_matcher.identify(phan_emb, threshold=0.363)
            print(f"  - Cross-identity check (Phanishree vs Alex): Identity='{cross_res['identity']}', Sim={cross_res['similarity']:.4f}")
            assert cross_res["identity"] == "Unknown", "Phanishree must NOT be recognized as Alex!"
            dark_test_res = alex_matcher.identify(cond_very_dark, threshold=0.363)
            print(f"  - Unrelated dark face (test.png 0.04x vs Alex): Identity='{dark_test_res['identity']}', Sim={dark_test_res['similarity']:.4f}")
            assert dark_test_res["identity"] == "Unknown", "Unrelated dark photo must NOT be recognized as Alex!"
            print("  ✓ Existing database cross-identity & unrelated-dark rejection verified")

    print("\n" + "=" * 70)
    print(" [SUCCESS] All Face Matching and Recognition tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()

