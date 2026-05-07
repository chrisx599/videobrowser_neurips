from __future__ import annotations

import json
from pathlib import Path
from typing import Any, ClassVar, Optional

import numpy as np

from videobrowser.memory.embedder import FrozenEmbedder
from videobrowser.memory.index import MemoryIndex
from videobrowser.search_engine.base import SearchMethodNotBuilt, register
from videobrowser.search_engine.pool import build_doc_text, compute_pool_fingerprint
from videobrowser.search_engine.schemas import ENGINE_VERSION, IndexMetadata, PoolRecord, SearchHit


def _default_embedder(model_name: str) -> FrozenEmbedder:
    from sentence_transformers import SentenceTransformer

    backend = SentenceTransformer(model_name)
    return FrozenEmbedder(backend)


@register
class EmbeddingRetriever:
    name: ClassVar[str] = "embedding"

    def __init__(self, embedder: Optional[FrozenEmbedder] = None, model_name: Optional[str] = None):
        self._index: Optional[MemoryIndex] = None
        self._embedder: Optional[FrozenEmbedder] = embedder
        self._model_name = model_name

    def _ensure_embedder(self) -> FrozenEmbedder:
        if self._embedder is not None:
            return self._embedder
        if not self._model_name:
            raise RuntimeError(
                "EmbeddingRetriever requires an embedder or a model_name (set via config or constructor)."
            )
        self._embedder = _default_embedder(self._model_name)
        return self._embedder

    def load(self, index_dir: Path) -> None:
        root = Path(index_dir) / self.name
        vectors_path = root / "vectors.npz"
        meta_path = root / "meta.json"
        if not vectors_path.exists() or not meta_path.exists():
            raise SearchMethodNotBuilt(self.name, index_dir)
        self._index = MemoryIndex.load(vectors_path)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if self._model_name is None:
            self._model_name = meta.get("extra", {}).get("model_name")

    def search(self, query: str, k: int) -> list[SearchHit]:
        if self._index is None or k <= 0:
            return []
        if not query or not query.strip():
            return []
        embedder = self._ensure_embedder()
        q_vec = embedder.embed_query(query)
        if q_vec.size == 0:
            return []
        top = self._index.top_n(q_vec, k)
        return [
            SearchHit(id=rec_id, score=float(score), method=self.name)
            for rec_id, score in top
            if float(score) > 0
        ]

    @classmethod
    def build(
        cls,
        records: list[PoolRecord],
        index_dir: Path,
        fields: list[str],
        model_name: str,
        batch_size: int = 32,
        embedder: Optional[FrozenEmbedder] = None,
        **_: Any,
    ) -> IndexMetadata:
        root = Path(index_dir) / cls.name
        root.mkdir(parents=True, exist_ok=True)

        texts: list[str] = []
        doc_ids: list[str] = []
        for rec in records:
            text = build_doc_text(rec, fields)
            if not text.strip():
                text = rec.title or (rec.id or "")
            texts.append(text)
            doc_ids.append(rec.id or "")

        embedder = embedder or _default_embedder(model_name)

        vectors: list[np.ndarray] = []
        for start in range(0, len(texts), max(batch_size, 1)):
            batch = texts[start : start + max(batch_size, 1)]
            batch_vecs = embedder.embed_texts(batch)
            if batch_vecs.size == 0:
                continue
            vectors.append(np.asarray(batch_vecs, dtype=float))

        if vectors:
            embeddings = np.concatenate(vectors, axis=0)
        else:
            embeddings = np.zeros((0, 0), dtype=float)

        index = MemoryIndex(memory_ids=doc_ids, embeddings=embeddings)
        index.save(root / "vectors.npz")
        (root / "doc_ids.json").write_text(
            json.dumps(doc_ids, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        fingerprint = compute_pool_fingerprint(records, fields)
        meta = IndexMetadata(
            method=cls.name,
            fingerprint=fingerprint,
            engine_version=ENGINE_VERSION,
            doc_count=len(records),
            fields=list(fields),
            extra={
                "model_name": model_name,
                "batch_size": batch_size,
                "dim": int(embeddings.shape[1]) if embeddings.ndim == 2 else 0,
                "normalized": True,
            },
        )
        (root / "meta.json").write_text(
            json.dumps(meta.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return meta
