from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from videobrowser.search_engine.schemas import PoolRecord
from videobrowser.utils.parser import extract_youtube_id


def _derive_id(record: PoolRecord) -> str:
    if record.id:
        return record.id
    if record.link:
        extracted = extract_youtube_id(record.link)
        if extracted:
            return extracted
        return hashlib.sha1(record.link.encode("utf-8")).hexdigest()[:16]
    raw = (record.title or "") + "|" + (record.description or "")
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def load_pool(path: str | Path) -> list[PoolRecord]:
    """Stream JSONL, model-validate each line, warn+skip malformed lines.

    Assigns a stable id when missing (derived from link or content hash).
    Deduplicates by id with last-write-wins and a warning.
    """
    pool_path = Path(path)
    if not pool_path.exists():
        raise FileNotFoundError(f"Pool file not found: {pool_path}")

    by_id: dict[str, PoolRecord] = {}
    with pool_path.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                payload = json.loads(raw)
                record = PoolRecord.model_validate(payload)
            except Exception as exc:
                print(f"⚠️ [OfflineSearch] Pool line {lineno} malformed, skipping: {exc}")
                continue

            rec_id = _derive_id(record)
            record.id = rec_id
            if rec_id in by_id:
                print(f"⚠️ [OfflineSearch] Duplicate id {rec_id!r} at line {lineno}; last write wins.")
            by_id[rec_id] = record

    return list(by_id.values())


def _transcript_to_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if "text" in value and isinstance(value["text"], str):
            return value["text"]
        try:
            return json.dumps(value, ensure_ascii=False)
        except Exception:
            return str(value)
    if isinstance(value, list):
        parts: list[str] = []
        for seg in value:
            if isinstance(seg, str):
                parts.append(seg)
            elif isinstance(seg, dict) and isinstance(seg.get("text"), str):
                parts.append(seg["text"])
            else:
                parts.append(str(seg))
        return " ".join(p for p in parts if p)
    return str(value)


def build_doc_text(record: PoolRecord, fields: Iterable[str]) -> str:
    """Concatenate indexed fields into a single searchable text blob."""
    data = record.model_dump()
    chunks: list[str] = []
    for field in fields:
        value = data.get(field)
        if value is None:
            continue
        if field == "transcript":
            text = _transcript_to_text(value)
        elif isinstance(value, list):
            text = " ".join(str(v) for v in value if v is not None)
        else:
            text = str(value)
        text = text.strip()
        if text:
            chunks.append(text)
    return "\n".join(chunks)


def build_field_texts(record: PoolRecord, fields: Iterable[str]) -> dict[str, str]:
    """Return per-field text (for retrievers that want field-level boost)."""
    data = record.model_dump()
    out: dict[str, str] = {}
    for field in fields:
        value = data.get(field)
        if value is None:
            out[field] = ""
            continue
        if field == "transcript":
            out[field] = _transcript_to_text(value).strip()
        elif isinstance(value, list):
            out[field] = " ".join(str(v) for v in value if v is not None).strip()
        else:
            out[field] = str(value).strip()
    return out


def compute_pool_fingerprint(records: list[PoolRecord], fields: Iterable[str]) -> str:
    """Stable fingerprint over (id, len(indexed_text)) pairs."""
    fields = list(fields)
    h = hashlib.sha256()
    for rec in sorted(records, key=lambda r: r.id or ""):
        text = build_doc_text(rec, fields)
        h.update((rec.id or "").encode("utf-8"))
        h.update(b"|")
        h.update(str(len(text)).encode("ascii"))
        h.update(b"\n")
    return h.hexdigest()
