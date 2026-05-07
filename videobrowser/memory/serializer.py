from __future__ import annotations

from typing import Any

from videobrowser.memory.schemas import RetrievalContext


def _video_status_counts(state: dict[str, Any]) -> dict[str, int]:
    counts = {
        "verified_video_count": 0,
        "watched_video_count": 0,
        "candidate_video_count": 0,
        "rejected_video_count": 0,
    }
    for video in state.get("video_store", {}).values():
        status = getattr(video, "status", None) or video.get("status")
        if status == "verified":
            counts["verified_video_count"] += 1
        elif status == "watched":
            counts["watched_video_count"] += 1
        elif status == "candidate":
            counts["candidate_video_count"] += 1
        elif status == "rejected":
            counts["rejected_video_count"] += 1
    return counts


def serialize_planner_state(state: dict[str, Any]) -> RetrievalContext:
    checker_state = state.get("checker_state", {})
    missing_slots = checker_state.get("missing_slots", [])
    query_text = state.get("user_query", "")
    if missing_slots:
        query_text = f"{query_text} Missing slots: {', '.join(missing_slots)}"
    return RetrievalContext(
        role="planner",
        query_text=query_text,
        state_features={
            "tried_query_count": len(state.get("tried_queries", [])),
            "loop_step": state.get("loop_step", 0),
            "missing_slots": list(missing_slots),
            **_video_status_counts(state),
        },
        filter_tags={},
    )
