from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field

ENGINE_VERSION = "1"


class PoolRecord(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: Optional[str] = None
    title: Optional[str] = None
    link: Optional[str] = None
    description: Optional[str] = None
    channel: Optional[str] = None
    upload_date: Optional[str] = None
    duration: Optional[str] = None
    tags: Optional[list[str]] = None
    thumbnail: Optional[str] = None
    transcript: Optional[Any] = None


class SearchHit(BaseModel):
    id: str
    score: float
    method: str
    matched_fields: list[str] = Field(default_factory=list)


class IndexMetadata(BaseModel):
    method: str
    fingerprint: str
    engine_version: str = ENGINE_VERSION
    doc_count: int
    fields: list[str] = Field(default_factory=list)
    extra: dict[str, Any] = Field(default_factory=dict)


def normalize_hit_to_candidate(record: PoolRecord, hit: SearchHit) -> dict:
    """Produce the exact candidate dict shape used by serper_search / youtube_search."""
    link = record.link or ""
    data = record.model_dump()
    tags = data.get("tags") or []
    description = data.get("description") or ""
    snippet = description or record.title or ""
    thumbnail = data.get("thumbnail") or ""
    channel = data.get("channel") or "unknown"
    date = data.get("upload_date") or "unknown"
    duration = data.get("duration") or "unknown"
    return {
        "title": record.title or "",
        "link": link,
        "snippet": snippet,
        "duration": duration,
        "imageurl": thumbnail,
        "videourl": link,
        "source": "Offline",
        "channel": channel,
        "date": date,
        "position": "unknown",
        "id": hit.id,
        "score": hit.score,
        "method": hit.method,
        "tags": tags,
    }
