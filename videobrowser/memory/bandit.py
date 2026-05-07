from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np


@dataclass
class LinUCBSelector:
    alpha: float
    feature_dim: int
    a_matrix: np.ndarray = field(init=False)
    b_vector: np.ndarray = field(init=False)
    theta: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.a_matrix = np.eye(self.feature_dim, dtype=float)
        self.b_vector = np.zeros(self.feature_dim, dtype=float)
        self.theta = np.zeros(self.feature_dim, dtype=float)

    def _normalize(self, features: np.ndarray) -> np.ndarray:
        vector = np.clip(np.asarray(features, dtype=float).reshape(-1), -1.0, 1.0)
        norm = np.linalg.norm(vector)
        if norm == 0.0:
            return vector
        return vector / norm

    def rank(self, feature_map: dict[str, np.ndarray]) -> list[tuple[str, float]]:
        a_inv = np.linalg.inv(self.a_matrix)
        ranked: list[tuple[str, float, float]] = []
        for memory_id, features in feature_map.items():
            vector = self._normalize(features)
            mean = float(self.theta @ vector)
            bonus = float(np.sqrt(vector @ a_inv @ vector))
            ranked.append((memory_id, mean + self.alpha * bonus, float(vector[0]) if vector.size else 0.0))
        ranked.sort(key=lambda item: (round(item[1], 12), item[2]), reverse=True)
        return [(memory_id, score) for memory_id, score, _ in ranked]

    def update(self, features: np.ndarray, reward: float) -> None:
        vector = self._normalize(features)
        self.a_matrix += np.outer(vector, vector)
        self.b_vector += reward * vector
        self.theta = np.linalg.solve(self.a_matrix, self.b_vector)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "alpha": self.alpha,
            "feature_dim": self.feature_dim,
            "a_matrix": self.a_matrix.tolist(),
            "b_vector": self.b_vector.tolist(),
            "theta": self.theta.tolist(),
        }
        output.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LinUCBSelector":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        selector = cls(alpha=float(payload["alpha"]), feature_dim=int(payload["feature_dim"]))
        selector.a_matrix = np.asarray(payload["a_matrix"], dtype=float)
        selector.b_vector = np.asarray(payload["b_vector"], dtype=float)
        if selector.a_matrix.shape != (selector.feature_dim, selector.feature_dim):
            raise ValueError("a_matrix shape does not match feature_dim")
        if selector.b_vector.shape != (selector.feature_dim,):
            raise ValueError("b_vector shape does not match feature_dim")
        selector.theta = np.linalg.solve(selector.a_matrix, selector.b_vector)
        return selector
