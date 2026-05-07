from __future__ import annotations

import json
import pickle
import re
from pathlib import Path
from typing import ClassVar

from videobrowser.search_engine.base import SearchMethodNotBuilt, register
from videobrowser.search_engine.pool import build_doc_text, compute_pool_fingerprint
from videobrowser.search_engine.schemas import ENGINE_VERSION, IndexMetadata, PoolRecord, SearchHit

TOKENIZER_VERSION = "1"

_CJK_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]"
)
_CJK_RUN_PATTERN = re.compile(
    r"[\u3400-\u4dbf\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\uac00-\ud7af]+"
)
_WORD_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def tokenize(text: str) -> list[str]:
    """Word tokens; for CJK runs, also append character bigrams.

    bge-m3 embeddings handle CJK natively; for BM25 we use a conservative
    char-bigram fallback so Chinese / Japanese queries still retrieve.
    """
    if not text:
        return []
    lowered = text.lower()
    tokens = _WORD_PATTERN.findall(lowered)
    if _CJK_PATTERN.search(lowered):
        for run in _CJK_RUN_PATTERN.findall(lowered):
            if len(run) >= 2:
                tokens.extend(run[i : i + 2] for i in range(len(run) - 1))
            else:
                tokens.append(run)
    return tokens


@register
class BM25Retriever:
    name: ClassVar[str] = "bm25"

    def __init__(self) -> None:
        self._model = None
        self._doc_ids: list[str] = []

    def load(self, index_dir: Path) -> None:
        root = Path(index_dir) / self.name
        model_path = root / "bm25.pkl"
        ids_path = root / "doc_ids.json"
        if not model_path.exists() or not ids_path.exists():
            raise SearchMethodNotBuilt(self.name, index_dir)
        with model_path.open("rb") as handle:
            self._model = pickle.load(handle)
        self._doc_ids = json.loads(ids_path.read_text(encoding="utf-8"))

    def search(self, query: str, k: int) -> list[SearchHit]:
        if self._model is None or not self._doc_ids or k <= 0:
            return []
        tokens = tokenize(query)
        if not tokens:
            return []
        scores = self._model.get_scores(tokens)
        if len(scores) == 0:
            return []
        import numpy as np

        order = np.argsort(scores)[::-1][:k]
        hits: list[SearchHit] = []
        for idx in order:
            score = float(scores[idx])
            if score <= 0:
                continue
            hits.append(SearchHit(id=self._doc_ids[int(idx)], score=score, method=self.name))
        return hits

    @classmethod
    def build(
        cls,
        records: list[PoolRecord],
        index_dir: Path,
        fields: list[str],
        k1: float = 1.5,
        b: float = 0.75,
        **_: object,
    ) -> IndexMetadata:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError(
                "rank_bm25 not installed. Run: uv pip install rank_bm25"
            ) from exc

        root = Path(index_dir) / cls.name
        root.mkdir(parents=True, exist_ok=True)

        tokenized_corpus: list[list[str]] = []
        doc_ids: list[str] = []
        for rec in records:
            text = build_doc_text(rec, fields)
            tokens = tokenize(text)
            if not tokens:
                tokens = ["<empty>"]
                print(
                    f"⚠️ [OfflineSearch] BM25: empty document for id={rec.id}; indexing placeholder token."
                )
            tokenized_corpus.append(tokens)
            doc_ids.append(rec.id or "")

        model = BM25Okapi(tokenized_corpus, k1=k1, b=b)

        with (root / "bm25.pkl").open("wb") as handle:
            pickle.dump(model, handle, protocol=pickle.HIGHEST_PROTOCOL)
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
            extra={"k1": k1, "b": b, "tokenizer_version": TOKENIZER_VERSION},
        )
        (root / "meta.json").write_text(
            json.dumps(meta.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return meta
