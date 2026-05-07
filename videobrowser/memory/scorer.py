"""UtilityScorer — a small MLP that predicts how useful a memory card will be
for a given query, trained on counterfactual flip labels.

Operates on frozen BGE-M3 embeddings; does NOT train the embedder or the base
LLM. ~600k parameters. Intended to be blended with the LLM critic's
zero-shot applicability via `hybrid_alpha` at inference time.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:  # allow import-only usage without torch
    torch = None  # type: ignore
    nn = None  # type: ignore


DEFAULT_QUERY_DIM = 1024
DEFAULT_CARD_DIM = 1024
DEFAULT_STATE_FEATURE_KEYS = [
    "loop_step", "raw_candidate_count", "current_query_count",
    "verified_video_count", "watched_video_count", "candidate_video_count",
    "rejected_video_count", "repeat_watch",
]


@dataclass
class ScorerMeta:
    query_dim: int = DEFAULT_QUERY_DIM
    card_dim: int = DEFAULT_CARD_DIM
    state_feature_keys: list[str] | None = None
    hidden1: int = 256
    hidden2: int = 128
    dropout: float = 0.2

    @property
    def state_dim(self) -> int:
        return len(self.state_feature_keys or DEFAULT_STATE_FEATURE_KEYS)

    @property
    def input_dim(self) -> int:
        return self.query_dim + self.card_dim + self.state_dim


def featurize_state(state_features: dict | None, keys: Sequence[str]) -> np.ndarray:
    """Extract a fixed-order numeric feature vector from state_features dict.

    Boolean values become 0.0/1.0; missing keys become 0.0; non-numeric falls
    back to 0.0. Length matches len(keys)."""
    out = np.zeros(len(keys), dtype=np.float32)
    if not state_features:
        return out
    for i, k in enumerate(keys):
        v = state_features.get(k)
        if isinstance(v, bool):
            out[i] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)):
            out[i] = float(v)
        # else: leave as 0.0
    return out


class UtilityScorer(nn.Module if nn is not None else object):
    """MLP utility predictor. Input: concat(query_emb, card_emb, state_feat).

    Output: scalar in [0, 1] via sigmoid. 1.0 = card likely helps this query,
    0.0 = card likely hurts.
    """

    def __init__(self, meta: ScorerMeta | None = None):
        if torch is None:
            raise ImportError("UtilityScorer requires torch")
        super().__init__()
        self.meta = meta or ScorerMeta(state_feature_keys=list(DEFAULT_STATE_FEATURE_KEYS))
        keys = self.meta.state_feature_keys or DEFAULT_STATE_FEATURE_KEYS
        self.state_feature_keys = list(keys)

        d_in = self.meta.input_dim
        d1 = self.meta.hidden1
        d2 = self.meta.hidden2
        p = self.meta.dropout

        self.net = nn.Sequential(
            nn.Linear(d_in, d1),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(d1, d2),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(d2, 1),
        )

    def forward(self, x):  # logits
        return self.net(x).squeeze(-1)

    def predict(self, x):  # probability (sigmoid)
        return torch.sigmoid(self.forward(x))

    def score_batch(
        self,
        query_emb: np.ndarray,
        card_embs: np.ndarray,
        state_feature_vec: np.ndarray,
    ) -> np.ndarray:
        """Score a batch of cards for a single query.

        Args:
            query_emb: [query_dim]
            card_embs: [n_cards, card_dim]
            state_feature_vec: [state_dim]

        Returns:
            [n_cards] array of scores in [0, 1].
        """
        self.eval()
        n = card_embs.shape[0]
        q = np.broadcast_to(query_emb, (n, query_emb.shape[-1]))
        s = np.broadcast_to(state_feature_vec, (n, state_feature_vec.shape[-1]))
        x = np.concatenate([q, card_embs, s], axis=-1).astype(np.float32)
        with torch.no_grad():
            t = torch.from_numpy(x)
            out = torch.sigmoid(self.forward(t)).cpu().numpy()
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "meta": {
                "query_dim": self.meta.query_dim,
                "card_dim": self.meta.card_dim,
                "state_feature_keys": self.state_feature_keys,
                "hidden1": self.meta.hidden1,
                "hidden2": self.meta.hidden2,
                "dropout": self.meta.dropout,
            },
        }, str(path))

    @classmethod
    def load(cls, path: str | Path) -> "UtilityScorer":
        if torch is None:
            raise ImportError("UtilityScorer.load requires torch")
        checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
        meta_dict = checkpoint["meta"]
        meta = ScorerMeta(
            query_dim=meta_dict["query_dim"],
            card_dim=meta_dict["card_dim"],
            state_feature_keys=list(meta_dict.get("state_feature_keys") or DEFAULT_STATE_FEATURE_KEYS),
            hidden1=meta_dict.get("hidden1", 256),
            hidden2=meta_dict.get("hidden2", 128),
            dropout=meta_dict.get("dropout", 0.2),
        )
        model = cls(meta)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model
