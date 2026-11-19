"""
Face detection module using OpenCV YuNet ONNX model.
Includes configurable confidence thresholds, aspect-ratio-preserving scaling,
and fallback contrast/padding preprocessing to robustly detect partially
occluded (e.g., masked or spectacled), low-light, and high-resolution faces.
"""

import os
from typing import Optional, Tuple
import cv2
import numpy as np


class FaceDetector:
    """
    Detects faces in images using OpenCV YuNet deep learning model.
    """

    def __init__(
        self,
        model_path: str = "models/face_detection_yunet.onnx",
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        max_dim: int = 1280,
        min_dim: int = 240,
    ):
        """
        Initializes the YuNet face detector.

        Args:
            model_path: Path to the pretrained YuNet ONNX model file.
            score_threshold: Confidence threshold to filter out low-confidence detections (0.0 to 1.0).
                             Default is 0.5 to handle partially occluded faces (masks/spectacles).
            nms_threshold: Non-Maximum Suppression threshold to suppress overlapping boxes.
            top_k: Maximum number of bounding boxes to retain before NMS.
            max_dim: Maximum image dimension (width or height) before proportional downscaling.
            min_dim: Minimum image dimension before proportional upscaling.

        Raises:
            FileNotFoundError: If the model file is not found at the specified path.
            ValueError: If thresholds are out of valid bounds.
            RuntimeError: If OpenCV fails to initialize the YuNet model.
        """
        if not (0.0 < score_threshold <= 1.0):
            raise ValueError(f"score_threshold must be in (0.0, 1.0], got {score_threshold}.")
        if not (0.0 <= nms_threshold <= 1.0):
            raise ValueError(f"nms_threshold must be in [0.0, 1.0], got {nms_threshold}.")

        self.model_path = os.path.abspath(model_path)
        self.score_threshold = float(score_threshold)
        self.nms_threshold = float(nms_threshold)
        self.top_k = int(top_k)
        self.max_dim = int(max_dim)
        self.min_dim = int(min_dim)

        # Validate model existence with a friendly error message
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(
                f"Face detection model not found at '{model_path}'.\n"
                f"Absolute path checked: '{self.model_path}'.\n"
                "Please ensure that 'face_detection_yunet.onnx' is placed in the 'models/' directory."
            )

        try:
            # OpenCV YuNet requires an initial input size (width, height);
            # placeholder (320, 320) is updated dynamically per image during detection.
            self.detector = cv2.FaceDetectorYN.create(
                model=self.model_path,
                config="",
                input_size=(320, 320),
                score_threshold=self.score_threshold,
                nms_threshold=self.nms_threshold,
                top_k=self.top_k,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to load YuNet model from '{self.model_path}': {e}"
            ) from e

    def set_score_threshold(self, score_threshold: float) -> None:
        """Updates the confidence threshold dynamically on the YuNet model."""
        if not (0.0 < score_threshold <= 1.0):
            raise ValueError(f"score_threshold must be in (0.0, 1.0], got {score_threshold}.")
        self.score_threshold = float(score_threshold)
        self.detector.setScoreThreshold(self.score_threshold)

    @staticmethod
    def _ensure_bgr_uint8(image: np.ndarray) -> np.ndarray:
        """Converts grayscale or 4-channel (BGRA) images safely into 3-channel uint8 BGR."""
        if image.dtype != np.uint8:
            image = np.clip(image, 0, 255).astype(np.uint8)

        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim == 3 and image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.ndim == 3 and image.shape[2] == 3:
            return np.ascontiguousarray(image)

        raise ValueError(f"Unsupported image shape for face detection: {image.shape}")

    @staticmethod
    def _resize_keep_aspect(
        image: np.ndarray,
        max_dim: int,
        min_dim: int = 0,
    ) -> Tuple[np.ndarray, float]:
        """
        Proportionally resizes an image without distorting facial aspect ratio.
        Returns (resized_image, scale_factor).
        """
        h, w = image.shape[:2]
        longest = max(h, w)
        scale = 1.0

        if max_dim > 0 and longest > max_dim:
            scale = max_dim / float(longest)
            interp = cv2.INTER_AREA
        elif min_dim > 0 and longest < min_dim:
            scale = min_dim / float(longest)
            interp = cv2.INTER_LINEAR
        else:
            return image, 1.0

        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))
        resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
        return resized, scale

    @staticmethod
    def _apply_clahe_lighting(bgr_image: np.ndarray) -> np.ndarray:
        """
        Applies gentle Contrast Limited Adaptive Histogram Equalization (CLAHE)
        on the Luminance (L) channel in LAB color space.
        Improves contrast for low-light or shadowed masked/spectacled faces
        without altering colors or geometric proportions.
        """
        lab = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        cl = clahe.apply(l_chan)
        merged = cv2.merge((cl, a_chan, b_chan))
        return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    def _run_yunet_raw(self, bgr_img: np.ndarray) -> Optional[np.ndarray]:
        """Updates input size to match bgr_img and executes YuNet detection."""
        h, w = bgr_img.shape[:2]
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(bgr_img)
        return faces

    def _postprocess_detections(
        self,
        faces: Optional[np.ndarray],
        orig_w: int,
        orig_h: int,
        scale: float = 1.0,
        pad_x: int = 0,
        pad_y: int = 0,
    ) -> Optional[np.ndarray]:
        """
        Maps bounding boxes and 5 landmarks back to original image coordinates,
        removes invalid/degenerate detections, applies NMS to prevent duplicates,
        and sorts faces from highest confidence to lowest.
        """
        if faces is None or len(faces) == 0:
            return None

        mapped = faces.astype(np.float32).copy()

        # 1. Undo padding offset in the scaled image space before unscaling
        if pad_x != 0 or pad_y != 0:
            mapped[:, 0] -= float(pad_x)
            mapped[:, 1] -= float(pad_y)
            for idx in range(4, 14, 2):
                mapped[:, idx] -= float(pad_x)
                mapped[:, idx + 1] -= float(pad_y)

        # 2. Undo scale factor on boxes (0:4) and landmarks (4:14)
        if scale != 1.0:
            mapped[:, :14] /= float(scale)

        # 3. Filter invalid or out-of-bounds boxes
        valid_rows = []
        boxes_for_nms = []
        scores_for_nms = []

        for row in mapped:
            x, y, bw, bh = row[0:4]
            score = float(row[14])

            # Reject degenerate dimensions
            if bw < 10 or bh < 10:
                continue

            # Ensure center of box falls inside the original image frame
            cx = x + bw * 0.5
            cy = y + bh * 0.5
            if cx < 0 or cx > orig_w or cy < 0 or cy > orig_h:
                continue

            # Clamp box boundaries slightly so box stays visible inside image
            clamped_x = max(0.0, x)
            clamped_y = max(0.0, y)
            clamped_w = min(bw - (clamped_x - x), orig_w - clamped_x)
            clamped_h = min(bh - (clamped_y - y), orig_h - clamped_y)
            if clamped_w < 10 or clamped_h < 10:
                continue

            row[0], row[1], row[2], row[3] = clamped_x, clamped_y, clamped_w, clamped_h
            valid_rows.append(row)
            boxes_for_nms.append([float(clamped_x), float(clamped_y), float(clamped_w), float(clamped_h)])
            scores_for_nms.append(score)

        if not valid_rows:
            return None

        # 4. Run Non-Maximum Suppression to guarantee no overlapping duplicate detections
        indices = cv2.dnn.NMSBoxes(
            boxes_for_nms,
            scores_for_nms,
            score_threshold=self.score_threshold,
            nms_threshold=self.nms_threshold,
        )

        if len(indices) == 0:
            return None

        selected_indices = [int(idx) for idx in np.array(indices).flatten()]
        filtered = np.array([valid_rows[i] for i in selected_indices], dtype=np.float32)

        # 5. Sort descending by confidence score so faces[0] is always primary
        order = np.argsort(-filtered[:, 14])
        return filtered[order]

    def detect_faces(
        self,
        image: np.ndarray,
        score_threshold: Optional[float] = None,
    ) -> Optional[np.ndarray]:
        """
        Detects faces in an input image (BGR format).

        Applies aspect-ratio-preserving scaling when images are very large or very small,
        and performs safe contrast/padding fallback retries if initial detection fails
        (e.g., for masked, spectacled, low-light, or tightly cropped faces).

        Args:
            image: Input image as a NumPy array (H x W x C, BGR format).
            score_threshold: Optional per-call confidence threshold override.

        Returns:
            NumPy array of shape (N, 15) containing face detection results sorted by
            descending confidence, or None if no faces are detected.

        Raises:
            ValueError: If the input image is None or invalid.
        """
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError(
                "Invalid input image provided to detect_faces. Expected a non-empty NumPy array."
            )

        # Temporarily apply custom threshold if provided
        original_threshold = self.score_threshold
        if score_threshold is not None and score_threshold != self.score_threshold:
            self.set_score_threshold(score_threshold)

        try:
            bgr_orig = self._ensure_bgr_uint8(image)
            orig_h, orig_w = bgr_orig.shape[:2]

            # ------------------------------------------------------------------
            # Pass 1: Standard detection with safe proportional resizing
            # ------------------------------------------------------------------
            proc_img, scale = self._resize_keep_aspect(
                bgr_orig, max_dim=self.max_dim, min_dim=self.min_dim
            )
            raw_faces = self._run_yunet_raw(proc_img)
            faces = self._postprocess_detections(raw_faces, orig_w, orig_h, scale=scale)
            if faces is not None and len(faces) > 0:
                return faces

            # ------------------------------------------------------------------
            # Pass 2 (Fallback): Resize large close-ups to 640px max dimension
            # (Large selfie/masked faces often exceed YuNet's single-scale stride)
            # ------------------------------------------------------------------
            if max(orig_h, orig_w) > 640:
                proc_640, scale_640 = self._resize_keep_aspect(bgr_orig, max_dim=640, min_dim=0)
                raw_640 = self._run_yunet_raw(proc_640)
                faces = self._postprocess_detections(raw_640, orig_w, orig_h, scale=scale_640)
                if faces is not None and len(faces) > 0:
                    return faces

            # ------------------------------------------------------------------
            # Pass 3 (Fallback): CLAHE contrast normalization for dark/occluded faces
            # ------------------------------------------------------------------
            clahe_img = self._apply_clahe_lighting(bgr_orig)
            clahe_scaled, scale_clahe = self._resize_keep_aspect(
                clahe_img, max_dim=800, min_dim=self.min_dim
            )
            raw_clahe = self._run_yunet_raw(clahe_scaled)
            faces = self._postprocess_detections(raw_clahe, orig_w, orig_h, scale=scale_clahe)
            if faces is not None and len(faces) > 0:
                return faces

            # ------------------------------------------------------------------
            # Pass 4 (Fallback): Add 15% border padding for tightly cropped portraits
            # ------------------------------------------------------------------
            base_for_pad, scale_pad = self._resize_keep_aspect(
                clahe_img, max_dim=800, min_dim=self.min_dim
            )
            ph, pw = base_for_pad.shape[:2]
            pad_y = max(16, int(round(ph * 0.15)))
            pad_x = max(16, int(round(pw * 0.15)))
            padded = cv2.copyMakeBorder(
                base_for_pad, pad_y, pad_y, pad_x, pad_x, borderType=cv2.BORDER_REPLICATE
            )
            raw_padded = self._run_yunet_raw(padded)
            # Unpad coordinates in the scaled space, then unscale back to original space
            faces = self._postprocess_detections(
                raw_padded,
                orig_w=orig_w,
                orig_h=orig_h,
                scale=scale_pad,
                pad_x=pad_x,
                pad_y=pad_y,
            )
            return faces

        finally:
            if self.score_threshold != original_threshold:
                self.set_score_threshold(original_threshold)

    def draw_faces(
        self,
        image: np.ndarray,
        faces: Optional[np.ndarray],
        box_color: Tuple[int, int, int] = (0, 255, 0),
        box_thickness: int = 2,
        draw_landmarks: bool = True,
    ) -> np.ndarray:
        """
        Draws green bounding boxes and detection confidence labels on the image.

        Args:
            image: Original image as a NumPy array (BGR format).
            faces: Array of detected faces of shape (N, 15), or None.
            box_color: BGR tuple for bounding box (default is Green: (0, 255, 0)).
            box_thickness: Line thickness for the bounding box.
            draw_landmarks: Whether to also draw the 5 facial landmark keypoints.

        Returns:
            A new annotated image copy with bounding boxes and labels drawn.
        """
        if image is None or not isinstance(image, np.ndarray):
            raise ValueError("Invalid input image provided to draw_faces.")

        # Create a copy so the original image is preserved
        annotated_image = image.copy()

        # If no faces were detected, return the copy untouched
        if faces is None or len(faces) == 0:
            return annotated_image

        for face in faces:
            # 1. Bounding box coordinates: x, y, width, height
            x, y, w, h = face[0:4].astype(int)

            # Ensure coordinates stay within safe bounds for drawing
            x = max(0, x)
            y = max(0, y)

            # 2. Confidence score at index 14
            confidence = float(face[14])
            label = f"{confidence * 100:.1f}%"

            # Draw green bounding box around detected face
            cv2.rectangle(
                annotated_image,
                (x, y),
                (x + w, y + h),
                box_color,
                thickness=box_thickness,
            )

            # Format and draw confidence label with background box for high visibility
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.55
            font_thickness = 1

            (text_w, text_h), baseline = cv2.getTextSize(
                label, font, font_scale, font_thickness
            )

            # Position label: place above bounding box, or inside if too close to top edge
            label_y = y - 6 if y - text_h - 6 > 0 else y + text_h + 6
            label_bg_top = label_y - text_h - 2
            label_bg_bottom = label_y + baseline

            # Draw background rectangle for text
            cv2.rectangle(
                annotated_image,
                (x, label_bg_top),
                (x + text_w + 4, label_bg_bottom),
                box_color,
                thickness=cv2.FILLED,
            )

            # Draw black text over the green background rectangle for contrast
            cv2.putText(
                annotated_image,
                label,
                (x + 2, label_y),
                font,
                font_scale,
                (0, 0, 0),
                thickness=font_thickness,
                lineType=cv2.LINE_AA,
            )

            # 3. Optional: Draw 5 facial landmarks (Right eye, Left eye, Nose, Right mouth corner, Left mouth corner)
            if draw_landmarks:
                landmarks = [
                    (int(face[4]), int(face[5])),    # Right eye
                    (int(face[6]), int(face[7])),    # Left eye
                    (int(face[8]), int(face[9])),    # Nose tip
                    (int(face[10]), int(face[11])),  # Right mouth corner
                    (int(face[12]), int(face[13])),  # Left mouth corner
                ]
                landmark_colors = [
                    (255, 0, 0),    # Blue (Right eye)
                    (0, 0, 255),    # Red (Left eye)
                    (0, 255, 0),    # Green (Nose)
                    (255, 0, 255),  # Magenta (Right mouth)
                    (0, 255, 255),  # Yellow (Left mouth)
                ]
                for (lx, ly), color in zip(landmarks, landmark_colors):
                    if lx > 0 and ly > 0:
                        cv2.circle(annotated_image, (lx, ly), radius=2, color=color, thickness=-1)

        return annotated_image
