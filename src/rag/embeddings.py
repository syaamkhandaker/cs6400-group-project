"""MiniLM embeddings from the Hugging Face hub.

``dense.model`` in ``configs/default.yaml`` names the model. ``build_cache.py``
embeds document chunks. The pipeline embeds a question with the same model.
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer


class HuggingFaceEmbedder:
    def __init__(self, model_name: str, batch_size: int = 64):
        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)
        dimension = getattr(self.model, "get_embedding_dimension", None)
        if dimension is None:
            dimension = self.model.get_sentence_embedding_dimension
        self.dim = int(dimension())

    @classmethod
    def from_config(cls, config: dict) -> HuggingFaceEmbedder:
        dense = config["dense"]
        return cls(dense["model"], batch_size=int(dense.get("batch_size", 64)))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def embed_query(self, text: str) -> np.ndarray:
        return self._encode([text])[0]

    def _encode(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        return np.ascontiguousarray(vectors, dtype=np.float32)
