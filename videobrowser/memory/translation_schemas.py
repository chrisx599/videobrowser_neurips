"""Pydantic schemas for v2c translation memory.

Defines the closed-set enums (`evidence_type`, `answer_type`), the visual
evidence signature `ZSignature`, and the alignment-graph rule container
`GGraph`. Both ZSignature and GGraph are persisted under
`MemoryCard.metadata.translation` and are also re-derived on the planner
side as `z_Q` (without `searchable_context`).
"""
from __future__ import annotations

from typing import Any, List, Literal

from pydantic import BaseModel, Field, field_validator


EVIDENCE_TYPES: tuple[str, ...] = (
    "object_property",
    "action",
    "scene_setting",
    "entity_identification",
    "on_screen_text",
    "spoken_content",
    "temporal_relation",
    "cross_source_fact",
)

ANSWER_TYPES: tuple[str, ...] = (
    "COLOR",
    "COUNT",
    "PERSON",
    "OBJECT",
    "LOCATION",
    "DATE_TIME",
    "TEXT",
    "ENTITY_NAME",
    "OTHER",
)

EvidenceType = Literal[
    "object_property",
    "action",
    "scene_setting",
    "entity_identification",
    "on_screen_text",
    "spoken_content",
    "temporal_relation",
    "cross_source_fact",
]

AnswerType = Literal[
    "COLOR",
    "COUNT",
    "PERSON",
    "OBJECT",
    "LOCATION",
    "DATE_TIME",
    "TEXT",
    "ENTITY_NAME",
    "OTHER",
]


_TEXT_WORD_CAP = 30


def _cap_words(s: str, cap: int) -> str:
    """Truncate to at most `cap` whitespace-separated words; preserves
    the original spacing inside that prefix. Empty string passes through."""
    words = s.split()
    if len(words) <= cap:
        return s.strip()
    return " ".join(words[:cap])


class ZSignature(BaseModel):
    evidence_type: EvidenceType
    answer_type: AnswerType
    visual_target: str = Field(default="", max_length=400)
    searchable_context: str = Field(default="", max_length=400)
    search_warning: str = Field(default="", max_length=400)

    @field_validator("evidence_type", "answer_type", mode="before")
    @classmethod
    def _strip_enum_fields(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.strip()
        return v

    @field_validator("visual_target", "searchable_context", "search_warning", mode="before")
    @classmethod
    def _strip_and_cap(cls, v: Any) -> str:
        if v is None:
            return ""
        if not isinstance(v, str):
            raise ValueError("text fields must be strings")
        return _cap_words(v.strip(), _TEXT_WORD_CAP)


class GRule(BaseModel):
    visual_slot: str = Field(min_length=1, max_length=200)
    search_by: List[str] = Field(default_factory=list)
    avoid_search: List[str] = Field(default_factory=list)
    verify_visually: List[str] = Field(default_factory=list)

    @field_validator("visual_slot", mode="before")
    @classmethod
    def _strip_slot(cls, v: Any) -> str:
        if not isinstance(v, str):
            raise ValueError("visual_slot must be a string")
        return v.strip()

    @field_validator("search_by", "avoid_search", "verify_visually", mode="before")
    @classmethod
    def _normalize_list(cls, v: Any) -> list[str]:
        if v is None:
            return []
        if not isinstance(v, list):
            raise ValueError("rule fields must be lists")
        out: list[str] = []
        for item in v:
            if isinstance(item, str) and item.strip():
                out.append(item.strip())
        return out[:8]


_G_RULE_CAP = 3


class GGraph(BaseModel):
    rules: List[GRule] = Field(default_factory=list)

    @field_validator("rules", mode="after")
    @classmethod
    def _cap_rule_count(cls, v: list[GRule]) -> list[GRule]:
        return v[:_G_RULE_CAP]


def parse_z_signature(obj: dict) -> ZSignature:
    """Validate a raw dict from the LLM into a ZSignature.
    Raises pydantic.ValidationError on any closed-set mismatch."""
    return ZSignature.model_validate(obj or {})


def parse_g_graph(obj: dict) -> GGraph:
    """Validate a raw dict from the LLM into a GGraph (rules array, capped at 3)."""
    if not obj:
        return GGraph(rules=[])
    return GGraph.model_validate(obj)
