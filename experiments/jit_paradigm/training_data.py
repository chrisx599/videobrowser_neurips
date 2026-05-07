from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from videobrowser.memory.schemas import ExperienceEvent


def _episode_id(row: dict[str, Any], final_state: dict[str, Any]) -> str:
    row_id = row.get("row_id", "unknown")
    return str(final_state.get("episode_id") or f"jit-row-{row_id}")


def _video_values(final_state: dict[str, Any]) -> list[Any]:
    return list((final_state.get("video_store") or {}).values())


def _video_attr(video: Any, key: str, default: Any = None) -> Any:
    if isinstance(video, dict):
        return video.get(key, default)
    return getattr(video, key, default)


def _parse_summary(summary_text: str | None) -> dict[str, Any]:
    if not summary_text:
        return {}
    try:
        payload = json.loads(summary_text)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass
    return {}


def _parse_summary_compact(summary_text: str | None) -> tuple[bool, int]:
    if not summary_text:
        return False, 0
    try:
        payload = json.loads(summary_text)
    except Exception:
        return False, 0
    if not isinstance(payload, dict):
        return False, 0
    return bool(payload.get("relevant")), len(payload.get("windows", []) or [])


def _relevant_videos(final_state: dict[str, Any]) -> list[Any]:
    relevant = []
    for video in _video_values(final_state):
        analysis = _parse_summary(_video_attr(video, "summary"))
        if analysis.get("relevant") is True:
            relevant.append(video)
    return relevant


def _has_useful_answer(final_state: dict[str, Any]) -> bool:
    answer = (final_state.get("final_answer") or "").strip()
    if not answer:
        return False
    return not answer.startswith("No relevant") and not answer.startswith("Error")


def _planner_memory_text(final_state: dict[str, Any]) -> str:
    thought = ""
    plan_trace = list(final_state.get("plan_trace") or [])
    if plan_trace:
        thought = plan_trace[-1]
    queries = list(final_state.get("tried_queries") or final_state.get("current_search_queries") or [])
    query_text = ", ".join(queries) if queries else "no search queries recorded"
    return f"{thought or 'Thought unavailable.'} Search queries used: {query_text}."


def _build_video_candidates(final_state: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = []
    for video in _video_values(final_state):
        title = _video_attr(video, "title", "")
        video_id = _video_attr(video, "video_id", "")
        status = _video_attr(video, "status", "")
        summary = _video_attr(video, "summary")
        relevant, n_windows = _parse_summary_compact(summary)
        candidates.append(
            {
                "video_id": video_id,
                "title": title,
                "status": status,
                "relevant": relevant,
                "n_windows": n_windows,
            }
        )
    return candidates


def _build_loops_from_step_log(
    raw_step_log: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """
    Partition raw_step_log records into per-loop dicts and extract the analyst record.

    Returns:
        loops:   list of dicts keyed by loop_index, each with planner/searcher/watcher/checker.
        analyst: the analyst record (or None), which typically runs once at the end.
    """
    loops: dict[int, dict[str, Any]] = {}
    analyst_record: dict[str, Any] | None = None

    for record in raw_step_log:
        role = record.get("role")
        loop_idx = int(record.get("loop", 0))

        if role == "analyst":
            analyst_record = record
            continue

        if loop_idx not in loops:
            loops[loop_idx] = {
                "loop_index": loop_idx,
                "planner": None,
                "searcher": None,
                "watcher": None,
                "checker": None,
            }

        if role in ("planner", "searcher", "watcher", "checker"):
            loops[loop_idx][role] = record

    # Sort by loop_index and return as list
    sorted_loops = [loops[k] for k in sorted(loops.keys())]
    return sorted_loops, analyst_record


def build_jit_training_trace(
    row: dict[str, Any],
    final_state: dict[str, Any],
    *,
    reflections: dict[str, str] | None = None,
) -> dict[str, Any]:
    trace_id = _episode_id(row, final_state)
    processed_videos = _video_values(final_state)

    planner_text = (reflections or {}).get("planner") or _planner_memory_text(final_state)

    raw_step_log = list(final_state.get("step_log") or [])
    loops, analyst_record = _build_loops_from_step_log(raw_step_log)

    return {
        "trace_id": trace_id,
        "episode_id": trace_id,
        "row_id": row.get("row_id"),
        "question": final_state.get("user_query") or row.get("question", ""),
        "ground_truth": row.get("answer"),
        "final_answer": final_state.get("final_answer"),
        "is_correct": final_state.get("is_correct"),
        "steps": [
            {
                "role": "planner",
                "phase_tag": "exploration",
                "gap_tags": [],
                "outcome": "success" if processed_videos else "failure",
                "memory_text": planner_text,
            },
        ],
        # Rich trajectory fields
        "loop_count": int(final_state.get("loop_step") or 0),
        "tried_queries": list(final_state.get("tried_queries") or []),
        "plan_trace": list(final_state.get("plan_trace") or []),
        "visited_video_ids": list(final_state.get("visited_video_ids") or []),
        "total_tokens": int((final_state.get("metrics") or {}).get("total_tokens", 0) or 0),
        "video_candidates": _build_video_candidates(final_state),
        # New per-loop step_log fields
        "loops": loops,
        "analyst": analyst_record,
        "raw_step_log": raw_step_log,
    }


def _planner_reward(final_state: dict[str, Any], row: dict[str, Any]) -> float:
    if not _has_useful_answer(final_state):
        return -1.0
    is_correct = final_state.get("is_correct", False)
    return 1.0 if is_correct else -0.5


def build_jit_experience_events(row: dict[str, Any], final_state: dict[str, Any]) -> list[ExperienceEvent]:
    episode_id = _episode_id(row, final_state)
    contexts = final_state.get("memory_context") or {}
    if not isinstance(contexts, dict):
        return []

    context = contexts.get("planner")
    if not isinstance(context, dict):
        return []
    selected_memory_ids = list(context.get("selected_memory_ids", []))
    if not selected_memory_ids:
        return []

    applicability = context.get("applicability_scores")
    uncertainty_components = context.get("uncertainty_components")
    return [
        ExperienceEvent(
            event_id=str(uuid.uuid4()),
            role="planner",
            query_text=context.get("retrieval_query_text") or final_state.get("user_query") or row.get("question", ""),
            state_features=dict(context.get("retrieval_state_features", {})),
            candidate_memory_ids=list(context.get("candidate_memory_ids", [])),
            selected_memory_ids=selected_memory_ids,
            selector_scores=dict(context.get("selector_scores", {})),
            checker_before={},
            checker_after={},
            local_reward=_planner_reward(final_state, row),
            cost={"total_tokens": final_state.get("metrics", {}).get("total_tokens", 0)},
            episode_id=episode_id,
            applicability_scores=dict(applicability) if isinstance(applicability, dict) else None,
            critic_confidence=context.get("critic_confidence"),
            critic_rationale=context.get("critic_rationale"),
            retrieval_uncertainty=context.get("retrieval_uncertainty"),
            uncertainty_components=dict(uncertainty_components) if isinstance(uncertainty_components, dict) else None,
            combined_confidence=context.get("combined_confidence"),
            recommended_action=context.get("recommended_action"),
            ranker_source=context.get("ranker_source"),
        )
    ]


def append_jsonl_record(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False))
        handle.write("\n")


def append_jit_trace(path: str | Path, trace: dict[str, Any]) -> None:
    append_jsonl_record(path, trace)


def append_jit_events(path: str | Path, events: list[ExperienceEvent]) -> None:
    for event in events:
        append_jsonl_record(path, event.model_dump(mode="json"))
