"""
Face Matching and Recognition module.
Provides:
- Cosine similarity computation between 128-D SFace embeddings.
- 1:1 Face Verification (comparing two face images or embeddings).
- 1:N Face Identification (matching a probe face against registered identities in FaceDatabase).
- Multi-photo aggregation strategies (max similarity or mean centroid).
- Configurable matching threshold with strict rejection of unknown identities.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from modules.detection import FaceDetector
from modules.feature_extraction import FaceFeatureExtractor
from modules.face_database import FaceDatabase


class FaceMatcher:
    """
    Performs 1:1 face verification and 1:N face identification using
    cosine similarity on SFace 128-dimensional embeddings.
    """

    # Default threshold for SFace cosine similarity.
    # In academic benchmarks (LFW/SphereFace), SFace uses ~0.363 as standard threshold.
    # Practitioners adjust between 0.30 and 0.45 depending on desired FAR vs FRR.
    DEFAULT_THRESHOLD = 0.363

    def __init__(
        self,
        database: Optional[FaceDatabase] = None,
        detector: Optional[FaceDetector] = None,
        extractor: Optional[FaceFeatureExtractor] = None,
        threshold: float = DEFAULT_THRESHOLD,
        aggregation: str = "hybrid",
    ):
        """
        Initializes the FaceMatcher.

        Args:
            database: An instance of FaceDatabase. If None, a new empty FaceDatabase is created.
            detector: An instance of FaceDetector. If None, a new FaceDetector is initialized.
            extractor: An instance of FaceFeatureExtractor. If None, a new FaceFeatureExtractor is initialized.
            threshold: Configurable cosine similarity threshold (default: 0.363).
            aggregation: Strategy for multi-photo identity comparison:
                         - 'hybrid' (default): Consensus score (0.70 * max_template_sim + 0.30 * centroid_sim)
                           when K >= 2 templates exist, balancing multi-condition recall with impostor resistance.
                         - 'max': Compares probe against every photo of the person and takes the maximum similarity.
                         - 'mean': Compares probe against the unit-normalized mean embedding of the person.

        Raises:
            ValueError: If threshold is not in the valid range (-1.0 to 1.0) or aggregation is invalid.
        """
        if not (-1.0 <= threshold <= 1.0):
            raise ValueError(f"Cosine threshold must be between -1.0 and 1.0, got {threshold}.")
        if aggregation not in ("hybrid", "max", "mean"):
            raise ValueError(
                f"Aggregation strategy must be 'hybrid', 'max', or 'mean', got '{aggregation}'."
            )

        self.threshold = float(threshold)
        self.aggregation = aggregation

        # Initialize or assign dependencies
        self.database = database if database is not None else FaceDatabase(auto_load=False)
        self._detector = detector
        self._extractor = extractor

    @property
    def detector(self) -> FaceDetector:
        """Lazy-loaded FaceDetector instance."""
        if self._detector is None:
            self._detector = FaceDetector()
        return self._detector

    @property
    def extractor(self) -> FaceFeatureExtractor:
        """Lazy-loaded FaceFeatureExtractor instance."""
        if self._extractor is None:
            self._extractor = FaceFeatureExtractor()
        return self._extractor

    @staticmethod
    def compute_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Computes the cosine similarity between two 128-dimensional embeddings.

        Formula:
            similarity = (u . v) / (||u||_2 * ||v||_2)

        Args:
            embedding1: First embedding array (1D shape (128,) or 2D (1, 128)).
            embedding2: Second embedding array (1D shape (128,) or 2D (1, 128)).

        Returns:
            Cosine similarity value as a float bounded in [-1.0, 1.0].

        Raises:
            TypeError: If inputs are not NumPy arrays.
            ValueError: If shapes are invalid or contain non-finite values (NaN / Inf).
        """
        if embedding1 is None or embedding2 is None:
            raise ValueError("Embeddings cannot be None.")
        if not isinstance(embedding1, np.ndarray) or not isinstance(embedding2, np.ndarray):
            raise TypeError("Both embeddings must be numpy.ndarray instances.")
        if embedding1.size != 128 or embedding2.size != 128:
            raise ValueError(
                f"Both embeddings must have 128 dimensions. Got {embedding1.size} and {embedding2.size}."
            )
        if not np.isfinite(embedding1).all() or not np.isfinite(embedding2).all():
            raise ValueError("Embeddings contain non-finite numerical values (NaN or Inf).")

        u = embedding1.flatten().astype(np.float32)
        v = embedding2.flatten().astype(np.float32)

        norm_u = float(np.linalg.norm(u))
        norm_v = float(np.linalg.norm(v))

        if norm_u == 0.0 or norm_v == 0.0:
            return 0.0

        dot_product = float(np.dot(u, v))
        similarity = dot_product / (norm_u * norm_v)

        # Clip within theoretical bounds [-1.0, 1.0] to prevent slight floating point overflow
        return float(np.clip(similarity, -1.0, 1.0))

    def _extract_embedding_from_input(self, face_input: Union[np.ndarray, Any]) -> Tuple[Optional[np.ndarray], str]:
        """
        Helper method to extract a 128-D embedding from either:
        - A precomputed 128-D embedding array (1D or 2D).
        - A raw BGR image containing a face.

        Returns:
            Tuple of (embedding or None, status_message).
        """
        if face_input is None:
            return None, "Input is None"

        if not isinstance(face_input, np.ndarray):
            return None, f"Expected numpy.ndarray, got {type(face_input).__name__}"

        # Case 1: Input is already a 128-dimensional embedding
        if face_input.size == 128 and face_input.ndim in (1, 2):
            if not np.isfinite(face_input).all():
                return None, "Embedding contains NaN or Inf values"
            return face_input.reshape(1, 128).astype(np.float32), "OK"

        # Case 2: Input is an image (3D array: H x W x C)
        if face_input.ndim == 3:
            faces = self.detector.detect_faces(face_input)
            if faces is None or len(faces) == 0:
                return None, "No face detected in the image"

            # Use primary detected face
            aligned_face = self.extractor.align_crop(face_input, faces[0])
            embedding = self.extractor.extract_feature(aligned_face)
            return embedding, "OK"

        return None, f"Unsupported input shape: {face_input.shape}"

    def verify(
        self,
        probe1: np.ndarray,
        probe2: np.ndarray,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        1:1 Face Verification: Determines whether two faces belong to the same person.

        Args:
            probe1: First face image (H x W x C) or precomputed 128-D embedding.
            probe2: Second face image (H x W x C) or precomputed 128-D embedding.
            threshold: Optional custom threshold; if None, self.threshold is used.

        Returns:
            Dictionary containing:
            - 'verified' (bool): True if similarity >= threshold, False otherwise.
            - 'similarity' (float): Cosine similarity score between -1.0 and 1.0.
            - 'threshold' (float): The threshold applied.
            - 'status' (str): Descriptive outcome message ('Match', 'Mismatch', or error reason).
        """
        active_threshold = float(threshold) if threshold is not None else self.threshold

        emb1, status1 = self._extract_embedding_from_input(probe1)
        if emb1 is None:
            return {
                "verified": False,
                "similarity": 0.0,
                "threshold": active_threshold,
                "status": f"Probe 1 failed: {status1}",
            }

        emb2, status2 = self._extract_embedding_from_input(probe2)
        if emb2 is None:
            return {
                "verified": False,
                "similarity": 0.0,
                "threshold": active_threshold,
                "status": f"Probe 2 failed: {status2}",
            }

        similarity = self.compute_cosine_similarity(emb1, emb2)
        is_match = similarity >= active_threshold

        return {
            "verified": bool(is_match),
            "similarity": float(similarity),
            "threshold": active_threshold,
            "status": "Match" if is_match else "Mismatch",
        }

    def identify(
        self,
        probe: np.ndarray,
        threshold: Optional[float] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        1:N Face Identification: Searches the FaceDatabase to identify the closest registered person.

        Args:
            probe: Probe face image (H x W x C) or precomputed 128-D embedding.
            threshold: Optional custom threshold; if None, self.threshold is used.
            top_k: Number of ranked candidate matches to return.

        Returns:
            Dictionary containing:
            - 'identity' (str): Name of matched person, or 'Unknown' if below threshold or database empty.
            - 'similarity' (float): Top match similarity score.
            - 'status' (str): Recognition status message ('Recognized', 'Unknown (Below threshold)', etc.).
            - 'threshold' (float): Threshold applied.
            - 'candidates' (List[Dict]): Ranked list of candidates [{'identity': name, 'similarity': score}, ...].
        """
        active_threshold = float(threshold) if threshold is not None else self.threshold

        # Handle empty database
        if len(self.database) == 0:
            return {
                "identity": "Unknown",
                "similarity": 0.0,
                "status": "Empty database: No identities enrolled",
                "threshold": active_threshold,
                "candidates": [],
            }

        # Extract embedding from probe
        probe_emb, probe_status = self._extract_embedding_from_input(probe)
        if probe_emb is None:
            return {
                "identity": "Unknown",
                "similarity": 0.0,
                "status": f"Probe failure: {probe_status}",
                "threshold": active_threshold,
                "candidates": [],
            }

        # Score probe against each identity in the database
        candidate_scores: List[Tuple[str, float]] = []

        for identity_name in self.database.get_identities():
            if self.aggregation == "mean":
                # Centroid matching: compare probe with normalized mean vector
                rep_emb = self.database.get_representative_embedding(identity_name)
                score = self.compute_cosine_similarity(probe_emb, rep_emb)
            elif self.aggregation == "max":
                # Max similarity matching: compare probe against each stored photo and pick the highest
                stored_embs = self.database.get_embeddings(identity_name)  # shape (K, 128)
                scores = [
                    self.compute_cosine_similarity(probe_emb, stored_embs[i])
                    for i in range(stored_embs.shape[0])
                ]
                score = max(scores)
            else:
                # Hybrid consensus matching (0.70 * best_template + 0.30 * centroid):
                # Allows multi-condition matching (e.g. low-light/spectacles templates)
                # while requiring centroid consensus to reject coincidental impostor matches.
                stored_embs = self.database.get_embeddings(identity_name)  # shape (K, 128)
                scores = [
                    self.compute_cosine_similarity(probe_emb, stored_embs[i])
                    for i in range(stored_embs.shape[0])
                ]
                max_score = max(scores)
                if len(scores) == 1:
                    score = max_score
                else:
                    rep_emb = self.database.get_representative_embedding(identity_name)
                    centroid_score = self.compute_cosine_similarity(probe_emb, rep_emb)
                    score = 0.70 * max_score + 0.30 * centroid_score

            candidate_scores.append((identity_name, float(score)))

        # Sort descending by similarity score
        candidate_scores.sort(key=lambda item: item[1], reverse=True)

        best_identity, best_score = candidate_scores[0]
        formatted_candidates = [
            {"identity": name, "similarity": round(score, 4)}
            for name, score in candidate_scores[:top_k]
        ]

        if best_score >= active_threshold:
            return {
                "identity": best_identity,
                "similarity": float(best_score),
                "status": "Recognized",
                "threshold": active_threshold,
                "candidates": formatted_candidates,
            }
        else:
            return {
                "identity": "Unknown",
                "similarity": float(best_score),
                "status": "Unknown (Below threshold)",
                "threshold": active_threshold,
                "candidates": formatted_candidates,
            }

