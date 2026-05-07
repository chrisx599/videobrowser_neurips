from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Optional

from videobrowser.search_engine.base import RETRIEVERS, SearchMethodNotBuilt, register
from videobrowser.search_engine.schemas import IndexMetadata, PoolRecord, SearchHit


def _rrf_fuse(
    ranked_lists: list[list[SearchHit]],
    k_rrf: int = 60,
    top_k: int = 10,
) -> list[SearchHit]:
    scores: dict[str, float] = {}
    matched: dict[str, set[str]] = {}
    for hits in ranked_lists:
        for rank, hit in enumerate(hits):
            scores[hit.id] = scores.get(hit.id, 0.0) + 1.0 / (k_rrf + rank + 1)
            matched.setdefault(hit.id, set()).add(hit.method)
    fused = sorted(scores.items(), key=lambda t: t[1], reverse=True)[:top_k]
    return [
        SearchHit(id=doc_id, score=score, method="hybrid", matched_fields=sorted(matched.get(doc_id, set())))
        for doc_id, score in fused
    ]


def _weighted_fuse(
    child_hits: dict[str, list[SearchHit]],
    weights: dict[str, float],
    top_k: int = 10,
) -> list[SearchHit]:
    scores: dict[str, float] = {}
    matched: dict[str, set[str]] = {}
    for method, hits in child_hits.items():
        if not hits:
            continue
        max_score = max((h.score for h in hits), default=1.0) or 1.0
        w = weights.get(method, 0.0)
        for hit in hits:
            normalized = (hit.score / max_score) if max_score else 0.0
            scores[hit.id] = scores.get(hit.id, 0.0) + w * normalized
            matched.setdefault(hit.id, set()).add(hit.method)
    fused = sorted(scores.items(), key=lambda t: t[1], reverse=True)[:top_k]
    return [
        SearchHit(id=doc_id, score=score, method="hybrid", matched_fields=sorted(matched.get(doc_id, set())))
        for doc_id, score in fused
    ]


@register
class HybridRetriever:
    name: ClassVar[str] = "hybrid"

    def __init__(
        self,
        children: Optional[list[str]] = None,
        fusion: str = "rrf",
        rrf_k: int = 60,
        weights: Optional[dict[str, float]] = None,
        embed_model_name: Optional[str] = None,
    ):
        self._children_names = children or ["bm25", "embedding"]
        self._fusion = fusion
        self._rrf_k = rrf_k
        self._weights = weights or {"bm25": 0.5, "embedding": 0.5}
        self._embed_model_name = embed_model_name
        self._children = []
        self._over_fetch = max(20, self._rrf_k)

    def load(self, index_dir: Path) -> None:
        instances = []
        for name in self._children_names:
            cls = RETRIEVERS.get(name)
            if cls is None or name == self.name:
                raise ValueError(f"Unknown or invalid hybrid child retriever: {name!r}")
            if name == "embedding":
                instance = cls(model_name=self._embed_model_name)
            else:
                instance = cls()
            instance.load(index_dir)
            instances.append((name, instance))
        self._children = instances

    def search(self, query: str, k: int) -> list[SearchHit]:
        if not self._children or k <= 0:
            return []
        over = max(k, self._over_fetch)
        per_method: dict[str, list[SearchHit]] = {}
        for name, instance in self._children:
            per_method[name] = instance.search(query, over)
        if self._fusion == "weighted":
            return _weighted_fuse(per_method, self._weights, top_k=k)
        return _rrf_fuse(list(per_method.values()), k_rrf=self._rrf_k, top_k=k)

    @classmethod
    def build(
        cls,
        records: list[PoolRecord],
        index_dir: Path,
        fields: list[str],
        **_: object,
    ) -> IndexMetadata:
        raise SearchMethodNotBuilt(
            cls.name,
            index_dir,
        )
