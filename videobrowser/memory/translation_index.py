"""Three-matrix index for v2c translation memory.

Mirrors `videobrowser/memory/index.py::MemoryIndex` but holds three parallel
embedding matrices keyed by `memory_ids`. The hybrid score per card is
`λ_q · cos(query_q, question_emb_i) + λ_z · cos(query_z_text, z_text_emb_i)
+ λ_t · cos(query_z_type, z_type_emb_i)`. All three λ are passed in by the
retriever (read from config); this class is a pure data container.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


def _row_cos(query: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Compute cosine similarity between a single query vector and each row
    of `matrix`. Returns an array of shape (matrix.shape[0],). Zero-norm
    rows or zero-norm queries yield zero similarity (no division-by-zero)."""
    if matrix.shape[0] == 0:
        return np.zeros((0,), dtype=float)
    q = np.asarray(query, dtype=float).reshape(-1)
    M = np.asarray(matrix, dtype=float)
    q_norm = np.linalg.norm(q)
    row_norms = np.linalg.norm(M, axis=1)
    denom = row_norms * q_norm
    return np.divide(M @ q, denom, out=np.zeros_like(row_norms, dtype=float), where=denom != 0)


@dataclass
class TranslationIndex:
    memory_ids: list[str]
    question_emb: np.ndarray
    z_text_emb: np.ndarray
    z_type_emb: np.ndarray

    def top_n(
        self,
        *,
        query_q: np.ndarray,
        query_z_text: np.ndarray,
        query_z_type: np.ndarray,
        n: int,
        lambda_q: float,
        lambda_z: float,
        lambda_t: float,
    ) -> list[tuple[str, float]]:
        if n <= 0 or len(self.memory_ids) == 0:
            return []
        scores = (
            lambda_q * _row_cos(query_q, self.question_emb)
            + lambda_z * _row_cos(query_z_text, self.z_text_emb)
            + lambda_t * _row_cos(query_z_type, self.z_type_emb)
        )
        order = np.argsort(scores)[::-1][:n]
        return [(self.memory_ids[i], float(scores[i])) for i in order]

    def get_embeddings(self, memory_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            idx = self.memory_ids.index(memory_id)
        except ValueError as exc:
            raise KeyError(f"memory_id {memory_id!r} not found in index") from exc
        return (
            np.asarray(self.question_emb[idx], dtype=float),
            np.asarray(self.z_text_emb[idx], dtype=float),
            np.asarray(self.z_type_emb[idx], dtype=float),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            np.savez(
                handle,
                memory_ids=np.asarray(self.memory_ids, dtype=str),
                question_emb=self.question_emb,
                z_text_emb=self.z_text_emb,
                z_type_emb=self.z_type_emb,
            )

    @classmethod
    def load(cls, path: str | Path) -> "TranslationIndex":
        with Path(path).open("rb") as handle, np.load(handle, allow_pickle=False) as data:
            memory_ids = data["memory_ids"].astype(str).tolist()
            question_emb = np.asarray(data["question_emb"], dtype=float)
            z_text_emb = np.asarray(data["z_text_emb"], dtype=float)
            z_type_emb = np.asarray(data["z_type_emb"], dtype=float)
        return cls(
            memory_ids=memory_ids,
            question_emb=question_emb,
            z_text_emb=z_text_emb,
            z_type_emb=z_type_emb,
        )
