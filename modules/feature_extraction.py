"""
Feature extraction module using OpenCV SFace (SphereFace/MobileFaceNet) ONNX model.
Includes controlled low-light illumination preprocessing (adaptive Gamma Correction
and gentle LAB CLAHE) and unit L2 embedding normalization so registration,
recognition, and verification share an identical, illumination-resilient pipeline.
"""

import os
from typing import Optional, Union
import cv2
import numpy as np


class FaceFeatureExtractor:
    """
    Aligns, crops, preprocesses under controlled lighting normalization, and extracts
    unit-normalized 128-dimensional facial feature embeddings using OpenCV SFace.
    """

    def __init__(
        self,
        model_path: str = "models/face_recognition_sface.onnx",
        normalize_illumination: bool = True,
        normalize_embedding: bool = True,
    ):
        """
        Initializes the SFace feature extractor.

        Args:
            model_path: Path to the pretrained SFace ONNX model file.
            normalize_illumination: If True, applies controlled adaptive gamma correction
                                    and LAB CLAHE to dark/overexposed aligned face crops
                                    while leaving well-lit faces unaltered.
            normalize_embedding: If True, L2-normalizes extracted 128-D embeddings to unit length.

        Raises:
            FileNotFoundError: If the model file is not found at the specified path.
            RuntimeError: If OpenCV fails to initialize the SFace model.
        """
        self.model_path = os.path.abspath(model_path)
        self.normalize_illumination = bool(normalize_illumination)
        self.normalize_embedding = bool(normalize_embedding)

        # Validate that the model file exists on disk
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"Face recognition model not found at '{model_path}'.\n"
                f"Absolute path checked: '{self.model_path}'.\n"
                "Please ensure that 'face_recognition_sface.onnx' is placed in the 'models/' directory."
            )

        try:
            # Create SFace recognizer instance
            self.recognizer = cv2.FaceRecognizerSF.create(
                model=self.model_path,
                config="",
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load SFace model from '{self.model_path}': {e}"
            ) from e

    @staticmethod
    def preprocess_aligned_face(aligned_face: np.ndarray) -> np.ndarray:
        """
        Applies controlled illumination preprocessing on a 112x112 aligned BGR face crop
        without mutating the input image or over-enhancing normal-light images.

        Strategy:
        - Convert crop to LAB color space and measure mean face luminance (L in [0, 255]).
        - Well-lit face (85 <= mean_L <= 195): preserved as-is to avoid unnecessary distortion
          and retain compatibility with standard portraits.
        - Low-light / very dark face (mean_L < 85):
          1. If extremely dark (mean_L < 45), apply a light edge-preserving bilateral filter
             (d=3) so boosting shadows does not amplify sensor grain.
          2. Compute smooth adaptive gamma = clip(log(0.45) / log(mean_L / 255), 0.22, 1.0)
             and lift dark intensities cleanly via LUT.
          3. Apply gentle CLAHE (clipLimit=2.0, tileGridSize=(4, 4)) strictly to the LAB L channel.
          4. Blend 75% CLAHE + 25% gamma-lifted image to prevent harsh local contrast artifacts.
        - Severely overexposed face (mean_L > 195): apply gentle highlight gamma recovery.

        Args:
            aligned_face: 112x112 BGR aligned face crop.

        Returns:
            A preprocessed copy of the aligned face crop (uint8 BGR).
        """
        if aligned_face is None or not isinstance(aligned_face, np.ndarray) or aligned_face.size == 0:
            raise ValueError("Invalid aligned face: expected a non-empty numpy.ndarray.")

        out = aligned_face.copy()
        if out.dtype != np.uint8:
            out = np.clip(out, 0, 255).astype(np.uint8)

        lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
        l_ch, _, _ = cv2.split(lab)
        mean_l = float(np.mean(l_ch))

        if mean_l < 85.0:
            # Mild edge-preserving denoise only on very dark inputs to prevent noise amplification
            if mean_l < 45.0:
                out = cv2.bilateralFilter(out, d=3, sigmaColor=20, sigmaSpace=20)

            norm_l = max(mean_l / 255.0, 0.01)
            gamma = float(np.clip(np.log(0.45) / np.log(norm_l), 0.22, 1.0))
            lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.uint8)
            gamma_img = cv2.LUT(out, lut)

            # Controlled CLAHE on Luminance channel only (preserves natural skin colors)
            g_lab = cv2.cvtColor(gamma_img, cv2.COLOR_BGR2LAB)
            gl, ga, gb = cv2.split(g_lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
            cl = clahe.apply(gl)
            clahe_img = cv2.cvtColor(cv2.merge((cl, ga, gb)), cv2.COLOR_LAB2BGR)

            # Soft blend avoids excessive local contrast enhancement
            out = cv2.addWeighted(clahe_img, 0.75, gamma_img, 0.25, 0)

        elif mean_l > 195.0:
            # Gentle highlight compression for severely overexposed inputs
            norm_l = min(mean_l / 255.0, 0.98)
            gamma = float(np.clip(np.log(0.55) / np.log(norm_l), 1.0, 1.6))
            lut = np.array([((i / 255.0) ** gamma) * 255.0 for i in range(256)], dtype=np.uint8)
            out = cv2.LUT(out, lut)

        return out

    def align_crop(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        """
        Aligns and crops a detected face based on its 5 facial landmarks.
        Normalizes head pose and scales face to 112x112 standard dimensions.

        Args:
            image: Original image as a NumPy array (H x W x C, BGR format).
            face: 1D or 2D NumPy array containing the 15-element YuNet detection result:
                  [0:4] bbox (x, y, w, h), [4:14] 5 landmarks (eyes, nose, mouth), [14] score.

        Returns:
            An aligned and cropped face image of size 112x112x3 (BGR format).

        Raises:
            ValueError: If image or face detection array is invalid or empty.
            TypeError: If arguments are not NumPy arrays.
            RuntimeError: If OpenCV alignCrop fails.
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Invalid input image: expected a non-empty numpy.ndarray.")

        if face is None:
            raise ValueError("Invalid face detection: 'face' cannot be None.")

        if not isinstance(face, np.ndarray):
            raise TypeError(f"Expected 'face' to be a numpy.ndarray, got {type(face).__name__}.")

        if face.size < 15:
            raise ValueError(
                f"Invalid face array: expected at least 15 elements (bbox + 5 landmarks + score), "
                f"but received size {face.size}."
            )

        # If a 2D array of shape (N, 15) was passed, take the first face row
        face_box = face[0] if face.ndim > 1 else face

        # Ensure contiguous float32 array for OpenCV C++ compatibility
        face_box = np.ascontiguousarray(face_box, dtype=np.float32)

        aligned_face = self.recognizer.alignCrop(image, face_box)

        if aligned_face is None or aligned_face.size == 0:
            raise RuntimeError("Face alignment failed: OpenCV alignCrop returned an empty result.")

        return aligned_face

    # Alias to support both snake_case and OpenCV's camelCase alignCrop
    alignCrop = align_crop

    def extract_feature(self, aligned_face: np.ndarray) -> np.ndarray:
        """
        Preprocesses (if enabled) and extracts a 128-dimensional numerical feature
        embedding vector from an aligned face.

        Args:
            aligned_face: 112x112 BGR aligned face crop (obtained from align_crop).

        Returns:
            A NumPy array of shape (1, 128) with dtype float32 representing the
            (unit-normalized) face embedding.

        Raises:
            ValueError: If the aligned face image is invalid.
            RuntimeError: If embedding extraction fails or contains non-finite values (NaN / Inf).
        """
        if aligned_face is None or not isinstance(aligned_face, np.ndarray) or aligned_face.size == 0:
            raise ValueError("Invalid aligned face: expected a non-empty numpy.ndarray.")

        # Apply identical controlled illumination preprocessing for both registration & recognition
        processed_face = (
            self.preprocess_aligned_face(aligned_face)
            if self.normalize_illumination
            else aligned_face
        )

        # Ensure contiguous array of shape (112, 112, 3)
        processed_face = np.ascontiguousarray(processed_face)

        # Extract 128-dimensional feature embedding
        embedding = self.recognizer.feature(processed_face)

        if embedding is None:
            raise RuntimeError("Failed to extract feature embedding from the aligned face.")

        # Verify that numerical values are finite (no NaN or Inf values)
        if not np.isfinite(embedding).all():
            raise RuntimeError("Extracted feature embedding contains non-finite values (NaN or Inf).")

        embedding = embedding.astype(np.float32).copy()

        # L2-normalize to unit length so multi-photo embeddings carry equal geometric weight
        if self.normalize_embedding:
            norm = float(np.linalg.norm(embedding))
            if norm > 0.0:
                embedding = embedding / norm

        return embedding

    # Alias to support both snake_case and OpenCV's feature method name
    feature = extract_feature

    def extract_from_face(self, image: np.ndarray, face: np.ndarray) -> np.ndarray:
        """
        Convenience method: performs alignment, cropping, preprocessing, and feature extraction.

        Args:
            image: Original image as a NumPy array (BGR format).
            face: 15-element face detection array from YuNet.

        Returns:
            A NumPy array of shape (1, 128) containing the normalized face embedding.
        """
        aligned_face = self.align_crop(image, face)
        return self.extract_feature(aligned_face)
