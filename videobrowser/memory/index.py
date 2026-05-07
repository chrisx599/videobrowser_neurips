from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MemoryIndex:
    memory_ids: list[str]
    embeddings: np.ndarray

    def top_n(self, query_embedding: np.ndarray, n: int) -> list[tuple[str, float]]:
        if n <= 0 or len(self.memory_ids) == 0:
            return []

        query = np.asarray(query_embedding, dtype=float).reshape(-1)
        embeddings = np.asarray(self.embeddings, dtype=float)
        query_norm = np.linalg.norm(query)
        row_norms = np.linalg.norm(embeddings, axis=1)
        denom = row_norms * query_norm
        scores = np.divide(
            embeddings @ query,
            denom,
            out=np.zeros_like(row_norms, dtype=float),
            where=denom != 0,
        )
        order = np.argsort(scores)[::-1][:n]
        return [(self.memory_ids[i], float(scores[i])) for i in order]

    def get_embedding(self, memory_id: str) -> np.ndarray:
        try:
            idx = self.memory_ids.index(memory_id)
        except ValueError:
            raise KeyError(f"memory_id {memory_id!r} not found in index")
        return np.asarray(self.embeddings[idx], dtype=float)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez(handle, memory_ids=np.asarray(self.memory_ids, dtype=str), embeddings=self.embeddings)

    @classmethod
    def load(cls, path: str | Path) -> "MemoryIndex":
        with Path(path).open("rb") as handle, np.load(handle, allow_pickle=False) as data:
            memory_ids = data["memory_ids"].astype(str).tolist()
            embeddings = np.asarray(data["embeddings"], dtype=float)
        return cls(memory_ids=memory_ids, embeddings=embeddings)
