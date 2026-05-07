from __future__ import annotations

from typing import Sequence

import numpy as np

from videobrowser.memory.schemas import MemoryCard


def _score_variance(scores: Sequence[float]) -> float:
    if len(scores) < 2:
        return 0.0
    arr = np.asarray(scores, dtype=float)
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo < 1e-9:
        return 0.0
    normed = (arr - lo) / (hi - lo)
    return float(np.var(normed))


def _outcome_entropy(cards: Sequence[MemoryCard]) -> float:
    if not cards:
        return 0.0
    successes = sum(1 for c in cards if c.outcome == "success")
    p = successes / len(cards)
    if p in (0.0, 1.0):
        return 0.0
    return float(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)))


def _embedding_spread(embeddings: Sequence[np.ndarray]) -> float:
    if embeddings is None or len(embeddings) < 2:
        return 0.0
    mat = np.stack([np.asarray(e, dtype=float).ravel() for e in embeddings], axis=0)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    normed = mat / norms
    sims = normed @ normed.T
    n = sims.shape[0]
    iu = np.triu_indices(n, k=1)
    distances = 1.0 - sims[iu]
    return float(np.clip(np.mean(distances), 0.0, 1.0))


def compute_retrieval_uncertainty(
    cards: Sequence[MemoryCard],
    scores: Sequence[float],
    embeddings: Sequence[np.ndarray] | None,
    weights: tuple[float, float, float] = (0.4, 0.3, 0.3),
) -> tuple[float, dict[str, float]]:
    """Combine score variance, outcome entropy, and embedding spread into a
    single scalar in [0, 1]. Returns (uncertainty, per-component dict)."""
    w_var, w_ent, w_spread = weights
    w_sum = max(w_var + w_ent + w_spread, 1e-9)

    score_var = _score_variance(scores)
    outcome_ent = _outcome_entropy(cards)
    emb_spread = _embedding_spread(embeddings) if embeddings else 0.0

    combined = (w_var * score_var + w_ent * outcome_ent + w_spread * emb_spread) / w_sum
    combined = float(np.clip(combined, 0.0, 1.0))
    return combined, {
        "score_variance": score_var,
        "outcome_entropy": outcome_ent,
        "embedding_spread": emb_spread,
    }
