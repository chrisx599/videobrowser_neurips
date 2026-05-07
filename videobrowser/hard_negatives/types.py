from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Optional

Verdict = Literal["YES", "NO", "UNCERTAIN"]


@dataclass(frozen=True)
class Candidate:
    """One YouTube candidate surfaced for a single benchmark question."""
    question_id: str
    video_id: str
    rewriter: str            # "qwen3" | "gpt54" | "gemini31"
    search_query: str
    yt_rank: int             # 1-indexed position in the YouTube search response
    title: Optional[str] = None
    channel: Optional[str] = None
    duration_seconds: Optional[int] = None
    filesize_bytes: Optional[int] = None
    description: Optional[str] = None


@dataclass(frozen=True)
class LayerVerdict:
    """Outcome of one verification layer for one (question, video)."""
    question_id: str
    video_id: str
    layer: Literal["layer1", "layer2"]
    verdict: Verdict
    reason: str
    prompt_path: Optional[str] = None  # filesystem ref for audit


@dataclass(frozen=True)
class Selection:
    """Final selected hard-negative for a question."""
    question_id: str
    video_id: str
    rewriter_overlap_score: int   # 1..3
    yt_rank_min: int              # best (smallest) yt_rank across rewriters
    layer1_verdict: Verdict
    layer2_verdict: Verdict
    source_query: str             # the search string we credit (the one with best yt_rank)
    source_rewriter: str
