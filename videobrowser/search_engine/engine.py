from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from videobrowser.search_engine.base import RETRIEVERS, SearchMethodNotBuilt
from videobrowser.search_engine.pool import load_pool
from videobrowser.search_engine.schemas import PoolRecord, SearchHit, normalize_hit_to_candidate

# Trigger retriever registration (side-effect imports)
from videobrowser.search_engine import keyword as _keyword  # noqa: F401
from videobrowser.search_engine import bm25 as _bm25  # noqa: F401
from videobrowser.search_engine import embedding as _embedding  # noqa: F401
from videobrowser.search_engine import hybrid as _hybrid  # noqa: F401


class OfflineSearchEngine:
    def __init__(
        self,
        pool_path: str | Path,
        index_dir: str | Path,
        default_method: str = "bm25",
        embed_model_name: Optional[str] = None,
        hybrid_children: Optional[list[str]] = None,
        hybrid_fusion: str = "rrf",
        hybrid_rrf_k: int = 60,
        hybrid_weights: Optional[dict[str, float]] = None,
    ):
        self.pool_path = Path(pool_path)
        self.index_dir = Path(index_dir)
        self.default_method = default_method
        self._embed_model_name = embed_model_name
        self._hybrid_kwargs = {
            "children": hybrid_children or ["bm25", "embedding"],
            "fusion": hybrid_fusion,
            "rrf_k": hybrid_rrf_k,
            "weights": hybrid_weights or {"bm25": 0.5, "embedding": 0.5},
            "embed_model_name": embed_model_name,
        }
        self._records: list[PoolRecord] = load_pool(self.pool_path)
        self._by_id: dict[str, PoolRecord] = {r.id: r for r in self._records if r.id}
        self._retrievers: dict[str, Any] = {}

    def _get_retriever(self, method: str):
        if method in self._retrievers:
            return self._retrievers[method]
        cls = RETRIEVERS.get(method)
        if cls is None:
            raise ValueError(f"Unknown offline search method: {method!r}")
        if method == "embedding":
            instance = cls(model_name=self._embed_model_name)
        elif method == "hybrid":
            instance = cls(**self._hybrid_kwargs)
        else:
            instance = cls()
        instance.load(self.index_dir)
        self._retrievers[method] = instance
        return instance

    def search(
        self,
        query: str,
        method: Optional[str] = None,
        k: int = 10,
        fields: Optional[list[str]] = None,
    ) -> list[dict]:
        m = method or self.default_method
        retriever = self._get_retriever(m)
        hits: list[SearchHit] = retriever.search(query, k)
        candidates: list[dict] = []
        for pos, hit in enumerate(hits, start=1):
            record = self._by_id.get(hit.id)
            if record is None:
                continue
            cand = normalize_hit_to_candidate(record, hit)
            cand["position"] = pos
            candidates.append(cand)
        return candidates


_DEFAULT_ENGINE_CACHE_KEY: Optional[tuple] = None
_DEFAULT_ENGINE: Optional[OfflineSearchEngine] = None


def _cache_key(config) -> tuple:
    offline = config.search.offline
    return (
        offline.enabled,
        offline.pool_path,
        offline.index_dir,
        offline.default.method,
        offline.embedding.model_name,
    )


def get_default_engine(config) -> Optional[OfflineSearchEngine]:
    """Return a shared engine instance for the given config, or None if disabled/unavailable."""
    global _DEFAULT_ENGINE, _DEFAULT_ENGINE_CACHE_KEY

    offline = config.search.offline
    if not offline.enabled:
        return None

    key = _cache_key(config)
    if _DEFAULT_ENGINE is not None and _DEFAULT_ENGINE_CACHE_KEY == key:
        return _DEFAULT_ENGINE

    try:
        engine = OfflineSearchEngine(
            pool_path=offline.pool_path,
            index_dir=offline.index_dir,
            default_method=offline.default.method,
            embed_model_name=offline.embedding.model_name,
            hybrid_children=offline.hybrid.children,
            hybrid_fusion=offline.hybrid.fusion,
            hybrid_rrf_k=offline.hybrid.rrf_k,
            hybrid_weights=offline.hybrid.weights,
        )
    except FileNotFoundError as exc:
        print(f"⚠️ [OfflineSearch] Pool file missing: {exc}")
        _DEFAULT_ENGINE = None
        _DEFAULT_ENGINE_CACHE_KEY = key
        return None
    except Exception as exc:
        print(f"⚠️ [OfflineSearch] Engine init failed: {exc}")
        _DEFAULT_ENGINE = None
        _DEFAULT_ENGINE_CACHE_KEY = key
        return None

    _DEFAULT_ENGINE = engine
    _DEFAULT_ENGINE_CACHE_KEY = key
    return engine


def reset_default_engine() -> None:
    global _DEFAULT_ENGINE, _DEFAULT_ENGINE_CACHE_KEY
    _DEFAULT_ENGINE = None
    _DEFAULT_ENGINE_CACHE_KEY = None
