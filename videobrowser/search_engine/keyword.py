from __future__ import annotations

import json
import re
from pathlib import Path
from typing import ClassVar

from videobrowser.search_engine.base import SearchMethodNotBuilt, register
from videobrowser.search_engine.pool import build_field_texts, compute_pool_fingerprint
from videobrowser.search_engine.schemas import ENGINE_VERSION, IndexMetadata, PoolRecord, SearchHit

FIELD_BOOST = {
    "title": 3.0,
    "tags": 2.0,
    "channel": 1.5,
    "description": 1.0,
    "transcript": 0.5,
}


@register
class KeywordRetriever:
    """Simple lowercase substring/regex match with per-field boost."""

    name: ClassVar[str] = "keyword"

    def __init__(self) -> None:
        self._docs: list[dict] = []
        self._fields: list[str] = []

    def load(self, index_dir: Path) -> None:
        root = Path(index_dir) / self.name
        docs_path = root / "docs.jsonl"
        meta_path = root / "meta.json"
        if not docs_path.exists() or not meta_path.exists():
            raise SearchMethodNotBuilt(self.name, index_dir)
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self._fields = meta.get("fields", [])
        self._docs = []
        with docs_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                self._docs.append(json.loads(line))

    def search(self, query: str, k: int, mode: str = "substring") -> list[SearchHit]:
        if not query or k <= 0 or not self._docs:
            return []

        tokens = [t for t in re.findall(r"\w+", query.lower(), flags=re.UNICODE) if t]
        if mode == "regex":
            pattern = re.compile(query, flags=re.IGNORECASE | re.UNICODE)
        else:
            if not tokens:
                return []
            pattern = re.compile(
                "|".join(re.escape(t) for t in tokens),
                flags=re.IGNORECASE | re.UNICODE,
            )

        scored: list[tuple[float, str, list[str]]] = []
        for doc in self._docs:
            total = 0.0
            matched: list[str] = []
            fields_map: dict[str, str] = doc["fields"]
            for field, text in fields_map.items():
                if not text:
                    continue
                hits = pattern.findall(text.lower() if mode != "regex" else text)
                if hits:
                    boost = FIELD_BOOST.get(field, 1.0)
                    total += boost * len(hits)
                    matched.append(field)
            if total > 0:
                scored.append((total, doc["id"], matched))

        scored.sort(key=lambda t: t[0], reverse=True)
        return [
            SearchHit(id=rec_id, score=float(score), method=self.name, matched_fields=mf)
            for score, rec_id, mf in scored[:k]
        ]

    @classmethod
    def build(
        cls,
        records: list[PoolRecord],
        index_dir: Path,
        fields: list[str],
        **_: object,
    ) -> IndexMetadata:
        root = Path(index_dir) / cls.name
        root.mkdir(parents=True, exist_ok=True)
        docs_path = root / "docs.jsonl"
        meta_path = root / "meta.json"

        with docs_path.open("w", encoding="utf-8") as handle:
            for rec in records:
                field_texts = build_field_texts(rec, fields)
                lowered = {k: v.lower() for k, v in field_texts.items()}
                handle.write(
                    json.dumps({"id": rec.id, "fields": lowered}, ensure_ascii=False) + "\n"
                )

        fingerprint = compute_pool_fingerprint(records, fields)
        meta = IndexMetadata(
            method=cls.name,
            fingerprint=fingerprint,
            engine_version=ENGINE_VERSION,
            doc_count=len(records),
            fields=list(fields),
            extra={"field_boost": FIELD_BOOST},
        )
        meta_path.write_text(json.dumps(meta.model_dump(), ensure_ascii=False, indent=2), encoding="utf-8")
        return meta
