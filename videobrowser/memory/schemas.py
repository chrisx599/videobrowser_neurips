from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


RoleName = Literal["planner", "watcher", "checker", "analyst"]
PhaseTag = Literal["exploration", "verification", "convergence"]
OutcomeTag = Literal["success", "failure"]


class MemoryCard(BaseModel):
    memory_id: str = Field(min_length=1)
    role: RoleName
    phase_tag: PhaseTag
    gap_tags: List[str] = Field(default_factory=list)
    outcome: OutcomeTag
    memory_text: str = Field(min_length=1)
    index_text: Optional[str] = Field(default=None, min_length=1)
    payload_text: Optional[str] = Field(default=None, min_length=1)
    tags: Dict[str, Any] = Field(default_factory=dict)
    source_trace_id: Optional[str] = None
    source_step_index: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CheckerState(BaseModel):
    answerable: bool
    current_stage: str
    missing_slots: List[str] = Field(default_factory=list)
    verified_slots: List[str] = Field(default_factory=list)
    drift: bool
    drift_type: Optional[str] = None
    confidence: Optional[float] = None
    signal: str = "planner"
    reason: str = ""
    missing_info: str = ""


class RetrievalContext(BaseModel):
    role: RoleName
    query_text: str
    state_features: Dict[str, Any] = Field(default_factory=dict)
    filter_tags: Dict[str, Any] = Field(default_factory=dict)
    proposal_top_n: int = 20
    selection_top_k: int = 3


class ExperienceEvent(BaseModel):
    event_id: str
    role: RoleName
    query_text: str
    state_features: Dict[str, Any] = Field(default_factory=dict)
    candidate_memory_ids: List[str] = Field(default_factory=list)
    selected_memory_ids: List[str] = Field(default_factory=list)
    selector_scores: Dict[str, float] = Field(default_factory=dict)
    checker_before: Dict[str, Any] = Field(default_factory=dict)
    checker_after: Dict[str, Any] = Field(default_factory=dict)
    local_reward: float = 0.0
    cost: Dict[str, Any] = Field(default_factory=dict)
    episode_id: Optional[str] = None
    applicability_scores: Optional[Dict[str, float]] = None
    critic_confidence: Optional[float] = None
    critic_rationale: Optional[str] = None
    retrieval_uncertainty: Optional[float] = None
    uncertainty_components: Optional[Dict[str, float]] = None
    combined_confidence: Optional[float] = None
    recommended_action: Optional[str] = None
    ranker_source: Optional[str] = None


class PreferenceSample(BaseModel):
    role: RoleName
    query_text: str
    state_features: Dict[str, Any] = Field(default_factory=dict)
    positive_memory_id: str
    negative_memory_id: str
    reward_delta: float
    label_source: str
