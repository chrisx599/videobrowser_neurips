from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, field_validator


_ACTION_VERBS = {
    "search", "rerank", "zoom", "broaden", "narrow",
    "switch", "re-query", "requery", "abstain", "look",
    "watch", "skip", "restart", "replace", "add",
}


class VideoSearchSkill(BaseModel):
    skill_id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=80)
    trigger_condition: str = Field(min_length=1)
    procedure: str = Field(min_length=1)
    applicable_categories: List[str] = Field(default_factory=list)
    supporting_episode_ids: List[str] = Field(min_length=2)
    success_rate: float = Field(ge=0.0, le=1.0)
    avg_token_saving: Optional[float] = None
    embedding: Optional[List[float]] = None
    source_model: str = Field(min_length=1)
    created_at: str = Field(min_length=1)

    @field_validator("procedure")
    @classmethod
    def _procedure_has_action_verb(cls, v: str) -> str:
        lowered = v.lower()
        if not any(verb in lowered for verb in _ACTION_VERBS):
            raise ValueError(
                "procedure must contain at least one action verb "
                f"from {sorted(_ACTION_VERBS)}"
            )
        return v
