"""
Robustness Evaluation module for Faceprint.
Evaluates face detection (YuNet) and face recognition (SFace) under challenging conditions:
- Bright lighting (bright_light)
- Low lighting (low_light)
- Side pose (side_pose)
- In-plane tilt (tilted)
- Blur (blurred)
- Gaussian / Sensor noise (noisy)
- Occlusion (occluded)

DISCLAIMER:
This module evaluates synthetic, AI-generated portrait variations for exploratory testing.
Results reflect pipeline resilience on stylized/synthetic variations, not real-world biometric accuracy.
"""

import os
import csv
from typing import Any, Dict, List, Optional, Tuple
import cv2
import numpy as np

from modules.detection import FaceDetector
from modules.feature_extraction import FaceFeatureExtractor
from modules.face_database import FaceDatabase
from modules.matching import FaceMatcher


class RobustnessEvaluator:
    """
    Automates enrollment of baseline reference images ('original.png') and
    benchmarks face detection and identification performance across 7 altered conditions.
    """

    ENROLLMENT_IMAGE_NAME = "original.png"
    ALTERED_CONDITIONS = [
        "bright_light",
        "low_light",
        "side_pose",
        "tilted",
        "blurred",
        "noisy",
        "occluded",
    ]

    def __init__(
        self,
        dataset_dir: str = "Faceprint_Test_Dataset",
        detector: Optional[FaceDetector] = None,
        extractor: Optional[FaceFeatureExtractor] = None,
        matcher: Optional[FaceMatcher] = None,
        threshold: float = 0.363,
    ):
        """
        Initializes the RobustnessEvaluator.

        Args:
            dataset_dir: Root path to the synthetic dataset directory.
            detector: FaceDetector instance (optional).
            extractor: FaceFeatureExtractor instance (optional).
            matcher: FaceMatcher instance (optional).
            threshold: Cosine similarity threshold for identification matching.
        """
        self.dataset_dir = self._resolve_dataset_root(dataset_dir)
        self.threshold = threshold

        # Initialize core pipeline modules
        self.detector = detector if detector is not None else FaceDetector()
        self.extractor = extractor if extractor is not None else FaceFeatureExtractor()

        # Use a fresh, dedicated in-memory FaceDatabase for the evaluation gallery
        self.database = FaceDatabase(auto_load=False)
        self.matcher = matcher if matcher is not None else FaceMatcher(
            database=self.database,
            detector=self.detector,
            extractor=self.extractor,
            threshold=self.threshold,
            aggregation="max",
        )
        # Ensure matcher points directly to our evaluation database
        self.matcher.database = self.database

        self.identities = self._discover_identities()

    def _resolve_dataset_root(self, base_dir: str) -> str:
        """
        Locates the dataset directory, automatically resolving nested subfolders if present.
        """
        abs_path = os.path.abspath(base_dir)

        # Check for nested subfolder: Faceprint_Test_Dataset/faceprint_synthetic_dataset
        nested_path = os.path.join(abs_path, "faceprint_synthetic_dataset")
        if os.path.isdir(nested_path):
            return nested_path

        if os.path.isdir(abs_path):
            return abs_path

        raise FileNotFoundError(
            f"Dataset directory not found at '{base_dir}'. "
            f"Expected folder with subject directories (e.g. Alex_Rivera, Sam_Taylor)."
        )

    def _discover_identities(self) -> List[str]:
        """
        Discovers all subject subdirectories containing an enrollment 'original.png' image.
        """
        if not os.path.isdir(self.dataset_dir):
            return []

        identities = []
        for entry in sorted(os.listdir(self.dataset_dir)):
            full_path = os.path.join(self.dataset_dir, entry)
            if os.path.isdir(full_path):
                enrollment_path = os.path.join(full_path, self.ENROLLMENT_IMAGE_NAME)
                if os.path.isfile(enrollment_path):
                    identities.append(entry)

        return identities

    def enroll_references(self) -> Dict[str, Dict[str, Any]]:
        """
        Enrolls the reference 'original.png' image for each discovered subject into the gallery.

        Returns:
            Dictionary mapping each subject to their enrollment status report.
        """
        enrollment_reports: Dict[str, Dict[str, Any]] = {}

        if not self.identities:
            raise RuntimeError(
                f"No subject folders with '{self.ENROLLMENT_IMAGE_NAME}' found in '{self.dataset_dir}'."
            )

        for identity in self.identities:
            image_path = os.path.join(self.dataset_dir, identity, self.ENROLLMENT_IMAGE_NAME)

            image = cv2.imread(image_path)
            if image is None:
                enrollment_reports[identity] = {
                    "success": False,
                    "reason": f"Failed to load image file at '{image_path}'",
                }
                continue

            # Detect face
            faces = self.detector.detect_faces(image)
            if faces is None or len(faces) == 0:
                enrollment_reports[identity] = {
                    "success": False,
                    "reason": "Face detection failed on reference image",
                }
                continue

            try:
                # Align and extract embedding
                aligned_face = self.extractor.align_crop(image, faces[0])
                embedding = self.extractor.extract_feature(aligned_face)

                # Register into local gallery
                self.database.register_face(identity, embedding, allow_append=True)

                enrollment_reports[identity] = {
                    "success": True,
                    "confidence": float(faces[0][14]),
                    "reason": "Enrolled successfully",
                }
            except Exception as e:
                enrollment_reports[identity] = {
                    "success": False,
                    "reason": f"Feature extraction/registration error: {e}",
                }

        return enrollment_reports

    def evaluate_probes(self) -> List[Dict[str, Any]]:
        """
        Tests each subject under all 7 altered conditions against the enrolled gallery.

        Returns:
            List of result dictionaries for every test condition sample.
        """
        results: List[Dict[str, Any]] = []

        if len(self.database) == 0:
            raise RuntimeError(
                "Database gallery is empty. Must run enroll_references() before evaluate_probes()."
            )

        for identity in self.identities:
            subject_folder = os.path.join(self.dataset_dir, identity)

            for condition in self.ALTERED_CONDITIONS:
                filename = f"{condition}.png"
                image_path = os.path.join(subject_folder, filename)

                record: Dict[str, Any] = {
                    "expected_identity": identity,
                    "condition": condition,
                    "detection_succeeded": False,
                    "recognition_succeeded": False,
                    "predicted_identity": "None",
                    "similarity_score": 0.0,
                    "failure_reason": "None",
                }

                # Check if probe image exists
                if not os.path.isfile(image_path):
                    record["failure_reason"] = f"Image file missing: '{filename}'"
                    results.append(record)
                    continue

                image = cv2.imread(image_path)
                if image is None:
                    record["failure_reason"] = f"Corrupt or unreadable image file: '{filename}'"
                    results.append(record)
                    continue

                # Step 1: Face Detection
                faces = self.detector.detect_faces(image)
                if faces is None or len(faces) == 0:
                    record["detection_succeeded"] = False
                    record["failure_reason"] = "Face detection failed (no face detected)"
                    results.append(record)
                    continue

                record["detection_succeeded"] = True

                # Step 2: Feature Extraction & Matching
                try:
                    aligned_face = self.extractor.align_crop(image, faces[0])
                    probe_embedding = self.extractor.extract_feature(aligned_face)

                    # 1:N Identification against enrolled gallery
                    match_result = self.matcher.identify(probe_embedding, threshold=self.threshold)
                    predicted_identity = match_result["identity"]
                    similarity = float(match_result["similarity"])

                    record["predicted_identity"] = predicted_identity
                    record["similarity_score"] = round(similarity, 4)

                    if predicted_identity == identity:
                        record["recognition_succeeded"] = True
                        record["failure_reason"] = "None (Matched correctly)"
                    elif predicted_identity == "Unknown":
                        record["recognition_succeeded"] = False
                        record["failure_reason"] = (
                            f"Below threshold (similarity {similarity:.4f} < {self.threshold:.3f})"
                        )
                    else:
                        record["recognition_succeeded"] = False
                        record["failure_reason"] = (
                            f"Misidentified as '{predicted_identity}' (similarity {similarity:.4f})"
                        )

                except Exception as e:
                    record["recognition_succeeded"] = False
                    record["failure_reason"] = f"Processing error during matching: {e}"

                results.append(record)

        return results

    def calculate_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Computes overall and per-condition detection success and correct identification rates.

        Returns:
            Dictionary containing aggregate summary metrics.
        """
        total_samples = len(results)
        if total_samples == 0:
            return {
                "total_samples": 0,
                "detection_success_rate": 0.0,
                "overall_identification_rate": 0.0,
                "conditional_identification_rate": 0.0,
                "per_condition": {},
            }

        total_detected = sum(1 for r in results if r["detection_succeeded"])
        total_recognized = sum(1 for r in results if r["recognition_succeeded"])

        detection_success_rate = (total_detected / total_samples) * 100.0
        overall_identification_rate = (total_recognized / total_samples) * 100.0
        conditional_identification_rate = (
            (total_recognized / total_detected) * 100.0 if total_detected > 0 else 0.0
        )

        # Per-condition breakdown
        per_condition: Dict[str, Dict[str, Any]] = {}
        for cond in self.ALTERED_CONDITIONS:
            cond_samples = [r for r in results if r["condition"] == cond]
            cond_total = len(cond_samples)
            if cond_total == 0:
                continue

            cond_detected = sum(1 for r in cond_samples if r["detection_succeeded"])
            cond_recognized = sum(1 for r in cond_samples if r["recognition_succeeded"])

            per_condition[cond] = {
                "total": cond_total,
                "detected": cond_detected,
                "detection_rate": round((cond_detected / cond_total) * 100.0, 1),
                "recognized": cond_recognized,
                "recognition_rate": round((cond_recognized / cond_total) * 100.0, 1),
                "avg_similarity": round(
                    float(np.mean([r["similarity_score"] for r in cond_samples if r["detection_succeeded"]]))
                    if cond_detected > 0 else 0.0,
                    4,
                ),
            }

        return {
            "total_samples": total_samples,
            "total_detected": total_detected,
            "detection_success_rate": round(detection_success_rate, 2),
            "total_recognized": total_recognized,
            "overall_identification_rate": round(overall_identification_rate, 2),
            "conditional_identification_rate": round(conditional_identification_rate, 2),
            "per_condition": per_condition,
        }

    def save_csv(
        self,
        results: List[Dict[str, Any]],
        output_path: str = "results/robustness_results.csv",
    ) -> str:
        """
        Saves detailed condition test results to a CSV file.

        Args:
            results: List of test result dictionaries.
            output_path: Destination path for CSV file.

        Returns:
            Absolute file path of the saved CSV.
        """
        abs_output = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(abs_output), exist_ok=True)

        fieldnames = [
            "expected_identity",
            "condition",
            "detection_succeeded",
            "recognition_succeeded",
            "predicted_identity",
            "similarity_score",
            "failure_reason",
        ]

        with open(abs_output, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                writer.writerow(row)

        return abs_output

    def run_full_evaluation(
        self,
        output_csv: str = "results/robustness_results.csv",
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Executes the end-to-end evaluation pipeline:
        1. Enrolls baseline references.
        2. Evaluates all 7 probe conditions.
        3. Computes summary metrics.
        4. Saves detailed results to CSV.

        Returns:
            Tuple of (detailed_results_list, summary_metrics_dict).
        """
        self.enroll_references()
        results = self.evaluate_probes()
        metrics = self.calculate_metrics(results)
        self.save_csv(results, output_path=output_csv)
        return results, metrics

