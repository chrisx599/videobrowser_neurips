from __future__ import annotations

from typing import Protocol

import numpy as np


class EmbeddingBackend(Protocol):
    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool = True,
    ) -> np.ndarray: ...


class FrozenEmbedder:
    def __init__(self, backend: EmbeddingBackend):
        self.backend = backend

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=float)
        return np.asarray(self.backend.encode(texts, normalize_embeddings=True), dtype=float)

    def embed_query(self, text: str) -> np.ndarray:
        embeddings = self.embed_texts([text])
        if embeddings.size == 0:
            return np.zeros((0,), dtype=float)
        return embeddings[0]
