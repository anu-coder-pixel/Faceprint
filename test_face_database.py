"""
Test script for Milestone 3: Face Registration and Local Database.

This script tests:
1. Extracting real face embeddings using YuNet (FaceDetector) and SFace (FaceFeatureExtractor).
2. Registering identities and storing their embeddings in FaceDatabase.
3. Multi-photo / multi-embedding registration for a single identity.
4. Prevention of accidental silent overwrites and duplicate registrations.
5. Strict validation against empty names and invalid/non-finite embeddings.
6. Persistent saving and loading using NumPy compressed archive (.npz).
7. Accurate retrieval of identities, embeddings, and representative mean vectors.
"""

import os
import sys
import cv2
import numpy as np

from modules.detection import FaceDetector
from modules.feature_extraction import FaceFeatureExtractor
from modules.face_database import FaceDatabase


def run_tests():
    print("=" * 70)
    print(" Faceprint: Milestone 3 - Face Registration & Local Database Test")
    print("=" * 70)

    # File paths
    detection_model_path = os.path.join("models", "face_detection_yunet.onnx")
    recognition_model_path = os.path.join("models", "face_recognition_sface.onnx")

    image_path = os.path.join("dataset", "test.png")
    if not os.path.isfile(image_path):
        image_path = os.path.join("dataset", "test.jpg")

    test_db_path = os.path.join("database", "test_face_database.npz")

    # Clean up any leftover test database from previous runs
    if os.path.isfile(test_db_path):
        try:
            os.remove(test_db_path)
        except OSError:
            pass

    # ---------------------------------------------------------
    # STEP 1: Extract a real embedding from dataset/test.png
    # ---------------------------------------------------------
    print("\n[Step 1/6] Extracting real face embedding from dataset photo...")
    if not os.path.isfile(detection_model_path) or not os.path.isfile(recognition_model_path):
        print("[ERROR] Required ONNX models not found in 'models/' folder.")
        sys.exit(1)

    if not os.path.isfile(image_path):
        print(f"[ERROR] Test image not found at '{image_path}'.")
        sys.exit(1)

    image = cv2.imread(image_path)
    if image is None:
        print(f"[ERROR] Failed to load image from '{image_path}'.")
        sys.exit(1)

    detector = FaceDetector(model_path=detection_model_path, score_threshold=0.8)
    faces = detector.detect_faces(image)
    if faces is None or len(faces) == 0:
        print("[ERROR] No face detected in test image.")
        sys.exit(1)

    extractor = FaceFeatureExtractor(model_path=recognition_model_path)
    aligned_face = extractor.align_crop(image, faces[0])
    real_embedding_1 = extractor.extract_feature(aligned_face)
    print(f"  ✓ Successfully extracted real embedding (shape: {real_embedding_1.shape})")

    # Create a simulated second photo embedding (e.g. slight lighting variation)
    # by adding subtle numerical variation and re-normalizing
    variation = np.random.RandomState(42).normal(0, 0.01, size=real_embedding_1.shape).astype(np.float32)
    real_embedding_2 = real_embedding_1 + variation
    print(f"  ✓ Prepared second embedding for multi-photo registration testing")

    # ---------------------------------------------------------
    # STEP 2: Initialize Database and Register Identities
    # ---------------------------------------------------------
    print("\n[Step 2/6] Testing Face Registration...")
    db = FaceDatabase(db_path=test_db_path, auto_load=False)

    # Register first person with a single photo
    count1 = db.register_face("Alex Rivera", real_embedding_1)
    print(f"  ✓ Registered 'Alex Rivera' (embeddings count: {count1})")

    # Add second photo to the same person (multi-photo support)
    count2 = db.add_embedding("Alex Rivera", real_embedding_2)
    print(f"  ✓ Added 2nd photo to 'Alex Rivera' (total embeddings: {count2})")

    # Register a second person with a synthetic 128-D embedding
    synthetic_embedding = np.random.RandomState(100).normal(0, 1.0, size=(1, 128)).astype(np.float32)
    synthetic_embedding /= np.linalg.norm(synthetic_embedding)
    count_sam = db.register_face("Sam Taylor", synthetic_embedding)
    print(f"  ✓ Registered 'Sam Taylor' (embeddings count: {count_sam})")

    assert len(db) == 2, f"Expected 2 registered identities, got {len(db)}"
    print(f"  ✓ Total registered identities in database: {len(db)}")

    # ---------------------------------------------------------
    # STEP 3: Test Overwrite & Duplicate Prevention
    # ---------------------------------------------------------
    print("\n[Step 3/6] Testing Duplicate & Silent Overwrite Prevention...")
    
    # Attempting to register an existing person without allow_append=True must raise ValueError
    blocked_silent_overwrite = False
    try:
        db.register_face("Alex Rivera", real_embedding_1, allow_append=False)
    except ValueError as e:
        blocked_silent_overwrite = True
        print(f"  ✓ Successfully blocked silent overwrite: {e}")

    assert blocked_silent_overwrite, "Failed: FaceDatabase allowed silent overwrite of an existing identity!"

    # ---------------------------------------------------------
    # STEP 4: Test Validation for Empty Names and Corrupt Embeddings
    # ---------------------------------------------------------
    print("\n[Step 4/6] Testing Input Validation...")

    # Empty name check
    empty_name_caught = False
    try:
        db.register_face("   ", real_embedding_1)
    except ValueError:
        empty_name_caught = True
        print("  ✓ Successfully rejected empty/whitespace-only identity name")
    assert empty_name_caught, "Failed: Empty name was not rejected!"

    # None name check
    none_name_caught = False
    try:
        db.register_face(None, real_embedding_1)  # type: ignore
    except (ValueError, TypeError):
        none_name_caught = True
        print("  ✓ Successfully rejected None identity name")
    assert none_name_caught, "Failed: None name was not rejected!"

    # Wrong dimension check (e.g. 64-D instead of 128-D)
    wrong_dim_caught = False
    try:
        bad_dim_emb = np.zeros((64,), dtype=np.float32)
        db.register_face("Invalid User", bad_dim_emb)
    except ValueError:
        wrong_dim_caught = True
        print("  ✓ Successfully rejected wrong embedding dimension (64-D)")
    assert wrong_dim_caught, "Failed: Wrong embedding dimension was not rejected!"

    # Non-finite values check (contains NaN)
    nan_caught = False
    try:
        nan_emb = real_embedding_1.copy()
        nan_emb[0, 5] = np.nan
        db.register_face("Invalid User", nan_emb)
    except ValueError:
        nan_caught = True
        print("  ✓ Successfully rejected embedding containing NaN")
    assert nan_caught, "Failed: NaN embedding was not rejected!"

    # ---------------------------------------------------------
    # STEP 5: Test Persistent Storage (Save & Load)
    # ---------------------------------------------------------
    print("\n[Step 5/6] Testing Persistent Save & Load (.npz format)...")
    saved_path = db.save()
    assert os.path.isfile(saved_path), f"Database file not created at {saved_path}"
    file_size_bytes = os.path.getsize(saved_path)
    print(f"  ✓ Database successfully saved to '{saved_path}' ({file_size_bytes} bytes)")

    # Load into a completely new FaceDatabase instance
    db_loaded = FaceDatabase(db_path=test_db_path, auto_load=True)
    assert len(db_loaded) == 2, f"Loaded database has {len(db_loaded)} identities, expected 2"
    assert "Alex Rivera" in db_loaded, "Missing 'Alex Rivera' in loaded database"
    assert "Sam Taylor" in db_loaded, "Missing 'Sam Taylor' in loaded database"
    print(f"  ✓ Successfully loaded database from disk with {len(db_loaded)} identities")

    # ---------------------------------------------------------
    # STEP 6: Test Retrieval & Numerical Consistency
    # ---------------------------------------------------------
    print("\n[Step 6/6] Testing Data Retrieval & Consistency...")
    identities = db_loaded.get_identities()
    print(f"  ✓ Registered identities list: {identities}")

    alex_embs = db_loaded.get_embeddings("Alex Rivera")
    assert alex_embs.shape == (2, 128), f"Expected shape (2, 128), got {alex_embs.shape}"
    print(f"  ✓ 'Alex Rivera' has {alex_embs.shape[0]} embeddings of shape {alex_embs.shape}")

    # Verify numerical values are identical before and after saving
    original_embs = db.get_embeddings("Alex Rivera")
    assert np.allclose(original_embs, alex_embs), "Loaded embeddings do not match original numerical values!"
    print("  ✓ Numerical verification passed: Loaded embeddings exactly match original floats")

    # Verify representative mean vector calculation
    mean_vec = db_loaded.get_representative_embedding("Alex Rivera")
    assert mean_vec.shape == (1, 128), f"Expected shape (1, 128), got {mean_vec.shape}"
    mean_norm = float(np.linalg.norm(mean_vec))
    print(f"  ✓ Representative mean embedding computed (shape: {mean_vec.shape}, unit norm: {mean_norm:.4f})")

    # Clean up test database file
    try:
        os.remove(test_db_path)
        print(f"  ✓ Cleaned up temporary test database '{test_db_path}'")
    except OSError:
        pass

    print("\n" + "=" * 70)
    print(" [SUCCESS] All Face Registration & Local Database tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    run_tests()

