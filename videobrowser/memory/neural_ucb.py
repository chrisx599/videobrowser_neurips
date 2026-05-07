from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


class NeuralUCBSelector(nn.Module):
    def __init__(
        self,
        emb_dim: int = 1024,
        hidden_dim: int = 128,
        output_dim: int = 64,
        alpha: float = 0.1,
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.update_count: int = 0

        self.query_proj = nn.Linear(emb_dim, hidden_dim)
        self.memory_proj = nn.Linear(emb_dim, hidden_dim)
        self.scorer = nn.Sequential(
            nn.Linear(hidden_dim * 3, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 1),
        )

    def forward(self, query_emb: torch.Tensor, memory_emb: torch.Tensor) -> torch.Tensor:
        q = self.query_proj(query_emb)
        m = self.memory_proj(memory_emb)
        combined = torch.cat([q, m, q * m], dim=-1)
        return self.scorer(combined).squeeze(-1)

    def _compute_grad_norm(self, query_t: torch.Tensor, memory_t: torch.Tensor) -> float:
        """Compute gradient norm of output w.r.t. parameters for exploration bonus."""
        for p in self.parameters():
            if p.grad is not None:
                p.grad = None
        query_t = query_t.detach().requires_grad_(False)
        memory_t = memory_t.detach().requires_grad_(False)
        score = self.forward(query_t.unsqueeze(0), memory_t.unsqueeze(0))
        score.backward()
        total_norm = 0.0
        for p in self.parameters():
            if p.grad is not None:
                total_norm += p.grad.norm().item() ** 2
        return total_norm ** 0.5

    def rank(self, query_embedding: np.ndarray, memory_embeddings: dict[str, np.ndarray]) -> list[tuple[str, float]]:
        if not memory_embeddings:
            return []

        self.eval()
        query_t = torch.as_tensor(np.asarray(query_embedding, dtype=np.float32)).reshape(-1)

        ranked: list[tuple[str, float]] = []
        t_decay = max(1.0, self.update_count) ** 0.5

        for memory_id, mem_emb in memory_embeddings.items():
            mem_t = torch.as_tensor(np.asarray(mem_emb, dtype=np.float32)).reshape(-1)

            with torch.no_grad():
                pred_reward = self.forward(query_t.unsqueeze(0), mem_t.unsqueeze(0)).item()

            grad_norm = self._compute_grad_norm(query_t, mem_t)
            bonus = self.alpha * grad_norm / t_decay
            ucb_score = pred_reward + bonus
            ranked.append((memory_id, ucb_score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def update(self, query_embedding: np.ndarray, memory_embedding: np.ndarray, reward: float, lr: float = 1e-3) -> None:
        self.train()
        query_t = torch.as_tensor(np.asarray(query_embedding, dtype=np.float32)).reshape(1, -1)
        memory_t = torch.as_tensor(np.asarray(memory_embedding, dtype=np.float32)).reshape(1, -1)
        target = torch.tensor([reward], dtype=torch.float32)

        pred = self.forward(query_t, memory_t)
        loss = nn.functional.mse_loss(pred, target)

        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        self.update_count += 1
        self.eval()

    def save(self, model_path: str | Path, meta_path: str | Path) -> None:
        model_path = Path(model_path)
        meta_path = Path(meta_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        meta_path.parent.mkdir(parents=True, exist_ok=True)

        torch.save(self.state_dict(), model_path)
        meta = {
            "emb_dim": self.emb_dim,
            "hidden_dim": self.hidden_dim,
            "output_dim": self.output_dim,
            "alpha": self.alpha,
            "update_count": self.update_count,
        }
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

    @classmethod
    def load(cls, model_path: str | Path, meta_path: str | Path) -> "NeuralUCBSelector":
        meta = json.loads(Path(meta_path).read_text(encoding="utf-8"))
        selector = cls(
            emb_dim=meta["emb_dim"],
            hidden_dim=meta["hidden_dim"],
            output_dim=meta["output_dim"],
            alpha=meta["alpha"],
        )
        selector.load_state_dict(torch.load(Path(model_path), weights_only=True))
        selector.update_count = meta.get("update_count", 0)
        selector.eval()
        return selector
