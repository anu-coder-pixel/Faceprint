"""
Face Database module for storing and managing registered face embeddings.
Supports multi-photo registration per identity, persistent NumPy (.npz) storage,
and robust validation against empty names, invalid embeddings, and accidental overwrites.
"""

import os
from typing import Dict, List, Optional, Union
import numpy as np


class FaceDatabase:
    """
    Manages local registration and persistent storage of facial feature embeddings.
    """

    def __init__(self, db_path: str = "database/face_database.npz", auto_load: bool = True):
        """
        Initializes the FaceDatabase.

        Args:
            db_path: Path to the local NumPy .npz database file.
            auto_load: If True and the database file exists, automatically load it.
        """
        self.db_path = os.path.abspath(db_path)
        # Internal storage: maps identity name (str) -> List of 1D NumPy arrays (each shape (128,))
        self.records: Dict[str, List[np.ndarray]] = {}

        if auto_load and os.path.isfile(self.db_path):
            self.load()

    def _validate_name(self, name: str) -> str:
        """
        Validates and cleans an identity name.

        Args:
            name: The person's name.

        Returns:
            Stripped name string.

        Raises:
            ValueError: If the name is None, empty, or consists only of whitespace.
            TypeError: If the name is not a string.
        """
        if name is None:
            raise ValueError("Identity name cannot be None.")
        if not isinstance(name, str):
            raise TypeError(f"Identity name must be a string, got {type(name).__name__}.")
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("Identity name cannot be empty or contain only whitespace.")
        return clean_name

    def _validate_embedding(self, embedding: np.ndarray) -> np.ndarray:
        """
        Validates that an embedding is a valid 128-dimensional finite numerical vector.

        Args:
            embedding: NumPy array representing a face embedding.

        Returns:
            A 1D NumPy array of shape (128,) with float32 dtype.

        Raises:
            TypeError: If embedding is not a NumPy array.
            ValueError: If shape is not 128-D or contains non-finite values (NaN / Inf).
        """
        if embedding is None:
            raise ValueError("Embedding cannot be None.")
        if not isinstance(embedding, np.ndarray):
            raise TypeError(f"Embedding must be a numpy.ndarray, got {type(embedding).__name__}.")
        if embedding.size != 128:
            raise ValueError(
                f"Invalid embedding dimension: expected 128 elements, got {embedding.size} "
                f"(shape: {embedding.shape})."
            )
        if not np.isfinite(embedding).all():
            raise ValueError("Embedding contains non-finite values (NaN or Inf).")

        # Flatten to 1D array of shape (128,) and float32 dtype, normalized to unit L2 length
        vec = np.ascontiguousarray(embedding.flatten(), dtype=np.float32)
        norm = float(np.linalg.norm(vec))
        if norm == 0.0:
            raise ValueError("Embedding cannot be a zero vector (L2 norm is 0).")
        return (vec / norm).astype(np.float32)

    def register_face(
        self,
        name: str,
        embedding: Union[np.ndarray, List[np.ndarray]],
        allow_append: bool = False,
    ) -> int:
        """
        Registers an identity with one or more face embeddings.

        Args:
            name: Person's name (e.g. "Jane Doe").
            embedding: A single 128-D embedding, or a list/2D-array of embeddings.
            allow_append: If True and the person is already registered, appends the new
                          embedding(s). If False and the person is already registered,
                          raises a ValueError to prevent silent overwriting.

        Returns:
            Total number of embeddings stored for this identity.

        Raises:
            ValueError: If name is empty, embedding is invalid, or identity already exists
                        and allow_append is False.
        """
        clean_name = self._validate_name(name)

        # Prevent accidental silent overwrite
        if clean_name in self.records and not allow_append:
            raise ValueError(
                f"Identity '{clean_name}' is already registered with {len(self.records[clean_name])} "
                f"embedding(s). Set allow_append=True or call add_embedding() to add more photos."
            )

        # Parse single or multiple embeddings
        embeddings_to_add: List[np.ndarray] = []
        if isinstance(embedding, list):
            if len(embedding) == 0:
                raise ValueError("Embedding list cannot be empty.")
            for emb in embedding:
                embeddings_to_add.append(self._validate_embedding(emb))
        elif isinstance(embedding, np.ndarray) and embedding.ndim == 2 and embedding.shape[0] > 1 and embedding.shape[1] == 128:
            # 2D array where each row is a 128-D embedding
            for i in range(embedding.shape[0]):
                embeddings_to_add.append(self._validate_embedding(embedding[i]))
        else:
            # Single embedding (1D or 1x128 2D)
            embeddings_to_add.append(self._validate_embedding(embedding))

        if clean_name not in self.records:
            self.records[clean_name] = []

        # Avoid adding near-exact duplicate embeddings for the same person
        existing_embeddings = self.records[clean_name]
        for new_emb in embeddings_to_add:
            is_duplicate = any(
                float(np.dot(new_emb, existing)) > 0.9999 or np.allclose(new_emb, existing, atol=1e-4)
                for existing in existing_embeddings
            )
            if not is_duplicate:
                existing_embeddings.append(new_emb)

        return len(self.records[clean_name])

    def add_embedding(self, name: str, embedding: np.ndarray) -> int:
        """
        Appends an additional face embedding to an already registered person.

        Args:
            name: Person's name.
            embedding: 128-D embedding to append.

        Returns:
            Total number of embeddings stored for this identity.

        Raises:
            ValueError: If name is not registered.
        """
        clean_name = self._validate_name(name)
        if clean_name not in self.records:
            raise ValueError(
                f"Identity '{clean_name}' is not registered. Use register_face() to register a new identity."
            )
        return self.register_face(clean_name, embedding, allow_append=True)

    def get_identities(self) -> List[str]:
        """Returns a list of all registered names sorted alphabetically."""
        return sorted(list(self.records.keys()))

    def get_embeddings(self, name: str) -> np.ndarray:
        """
        Retrieves all embeddings for a registered person.

        Args:
            name: Person's name.

        Returns:
            NumPy array of shape (K, 128) containing all K embeddings for that identity.

        Raises:
            KeyError: If identity is not found.
        """
        clean_name = self._validate_name(name)
        if clean_name not in self.records:
            raise KeyError(f"Identity '{clean_name}' not found in database.")
        return np.vstack(self.records[clean_name])

    def get_representative_embedding(self, name: str) -> np.ndarray:
        """
        Calculates the mean embedding across all registered photos for an identity.
        Normalizes the resulting vector to unit length for accurate matching.

        Args:
            name: Person's name.

        Returns:
            NumPy array of shape (1, 128) representing the averaged identity embedding.

        Raises:
            KeyError: If identity is not found.
        """
        embeddings = self.get_embeddings(name)  # shape (K, 128)
        mean_vec = np.mean(embeddings, axis=0, keepdims=True)  # shape (1, 128)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            mean_vec = mean_vec / norm
        return mean_vec.astype(np.float32)

    def get_all_records(self) -> Dict[str, np.ndarray]:
        """
        Returns a dictionary mapping each registered name to their (K, 128) NumPy array of embeddings.
        """
        return {name: np.vstack(embs) for name, embs in self.records.items()}

    def remove_identity(self, name: str) -> bool:
        """
        Removes an identity and all their embeddings from the database.

        Args:
            name: Person's name to remove.

        Returns:
            True if removed, False if identity was not found.
        """
        clean_name = self._validate_name(name)
        if clean_name in self.records:
            del self.records[clean_name]
            return True
        return False

    def save(self, db_path: Optional[str] = None) -> str:
        """
        Saves the database to a persistent NumPy compressed archive (.npz).

        Args:
            db_path: Target file path (optional; defaults to self.db_path).

        Returns:
            The absolute path where the database was saved.

        Raises:
            IOError: If saving fails.
        """
        save_path = os.path.abspath(db_path) if db_path else self.db_path
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Prepare dictionary of 2D NumPy arrays: {name: np.ndarray(K, 128)}
        data_to_save = {name: np.vstack(embs) for name, embs in self.records.items()}

        try:
            np.savez_compressed(save_path, **data_to_save)
        except Exception as e:
            raise IOError(f"Failed to save face database to '{save_path}': {e}") from e

        return save_path

    def load(self, db_path: Optional[str] = None) -> int:
        """
        Loads the database from a persistent NumPy compressed archive (.npz).

        Args:
            db_path: Source file path (optional; defaults to self.db_path).

        Returns:
            Total number of registered identities loaded.

        Raises:
            FileNotFoundError: If the database file does not exist.
            IOError: If loading fails.
        """
        load_path = os.path.abspath(db_path) if db_path else self.db_path

        if not os.path.isfile(load_path):
            raise FileNotFoundError(f"Face database file not found at '{load_path}'.")

        try:
            loaded_data = np.load(load_path, allow_pickle=False)
            new_records: Dict[str, List[np.ndarray]] = {}

            for name in loaded_data.files:
                embs_array = loaded_data[name]  # shape (K, 128)
                if embs_array.ndim == 1:
                    embs_array = embs_array.reshape(1, -1)

                new_records[name] = [
                    self._validate_embedding(embs_array[i]) for i in range(embs_array.shape[0])
                ]

            self.records = new_records
            return len(self.records)
        except Exception as e:
            raise IOError(f"Failed to load face database from '{load_path}': {e}") from e

    def count_total_embeddings(self) -> int:
        """Returns the total number of embeddings across all registered persons."""
        return sum(len(embs) for embs in self.records.values())

    def __len__(self) -> int:
        """Returns the number of unique registered identities."""
        return len(self.records)

    def __contains__(self, name: str) -> bool:
        """Checks if a person's name is registered."""
        try:
            clean_name = self._validate_name(name)
            return clean_name in self.records
        except (ValueError, TypeError):
            return False

    def __repr__(self) -> str:
        return (
            f"<FaceDatabase(identities={len(self.records)}, "
            f"total_embeddings={self.count_total_embeddings()}, path='{self.db_path}')>"
        )

