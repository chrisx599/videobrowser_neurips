import numpy as np
from videobrowser.config import get_config
from videobrowser.core.state import AgentState
from videobrowser.memory.bootstrap import get_default_memory_runtime
from videobrowser.memory.logging import append_event
from videobrowser.memory.reward import compute_local_reward
from videobrowser.memory.schemas import CheckerState, ExperienceEvent
from videobrowser.utils.llm_factory import get_llm
from videobrowser.utils.logger import get_logger
from videobrowser.utils.metrics import update_token_metrics
from videobrowser.utils.parser import extract_json_from_text
from videobrowser.utils.prompt_manager import load_prompt
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
import uuid

load_dotenv()

llm = get_llm(node_name="checker")
logger = get_logger()

REQUIRED_CHECKER_FIELDS = {
    "answerable",
    "current_stage",
    "missing_slots",
    "verified_slots",
    "confidence",
    "signal",
    "reason",
    "missing_info",
}


def _coerce_checker_state(payload: dict | None) -> CheckerState:
    base_payload = {
        "answerable": False,
        "current_stage": "exploration",
        "missing_slots": [],
        "verified_slots": [],
        "drift": False,
        "drift_type": None,
        "confidence": None,
        "signal": "planner",
        "reason": "",
        "missing_info": "",
    }
    if payload:
        base_payload.update(payload)
    return CheckerState(**base_payload)


def _parse_checker_state(payload: dict) -> CheckerState:
    missing_fields = sorted(REQUIRED_CHECKER_FIELDS - set(payload.keys()))
    if missing_fields:
        raise ValueError(f"Missing checker fields: {', '.join(missing_fields)}")
    return CheckerState.model_validate(payload)


def _fallback_checker_state(previous_checker_state: CheckerState) -> CheckerState:
    return _coerce_checker_state(
        {
            "current_stage": previous_checker_state.current_stage,
            "missing_slots": list(previous_checker_state.missing_slots),
            "verified_slots": list(previous_checker_state.verified_slots),
            "drift": True,
            "drift_type": "llm_error",
            "signal": "planner",
            "reason": "LLM error",
            "missing_info": "Retry search",
        }
    )


def checker_node(state: AgentState):
    print("🧐 [Checker] Evaluating evidence sufficiency...")

    config = get_config()
    max_loop_steps = config.checker.max_loop_steps

    current_step = state.get("loop_step", 0) + 1
    user_query = state["user_query"]
    previous_checker_state = _coerce_checker_state(state.get("checker_state", {}))

    logger.log("Checker", "start", {"current_step": current_step, "max_steps": max_loop_steps})

    # 1. Collect Verified Evidence
    verified_videos = [
        v for v in state["video_store"].values() if v.status == "verified"
    ]

    evidence_summary = ""
    for i, vid in enumerate(verified_videos):
        evidence_summary += f"\n[Video {i+1}: {vid.title}]\n"
        for frag in vid.evidence:
            evidence_summary += f"- {frag.content}\n"

    # 2. Evaluate via LLM
    prompt_text = load_prompt(
        "checker_evaluate.j2",
        user_query=user_query,
        evidence_summary=evidence_summary,
    )

    metrics = state.get("metrics", {})
    checker_state = previous_checker_state
    try:
        response = llm.invoke([HumanMessage(content=prompt_text)])
        metrics = update_token_metrics(metrics, response, category="checker")
        result = extract_json_from_text(response.content)
        checker_state = _parse_checker_state(result)

        logger.log("Checker", "decision", {
            "signal": checker_state.signal,
            "reason": checker_state.reason,
            "missing_info": checker_state.missing_info,
            "answerable": checker_state.answerable,
            "missing_slots": checker_state.missing_slots,
            "verified_slots": checker_state.verified_slots,
        })
    except Exception as e:
        print(f"   ⚠️ Checker LLM failed: {e}. Defaulting to Planner.")
        logger.log("Checker", "error", {"error": str(e)}, level="ERROR")
        checker_state = _fallback_checker_state(previous_checker_state)

    # 3. Force finish on max loops after parsing structured state
    if current_step >= max_loop_steps:
        print(f"   -> 🛑 Max loops ({max_loop_steps}) reached. Forcing finish.")
        logger.log("Checker", "max_loops_reached", {"steps": current_step})
        checker_state = checker_state.model_copy(
            update={
                "signal": "analyst",
                "reason": checker_state.reason or "Max loops reached. Proceeding with partial information.",
            }
        )

    runtime_flags = {
        "repeat_query": len(set(state.get("tried_queries", []))) < len(state.get("tried_queries", []))
        if state.get("tried_queries")
        else False,
        "repeat_watch": len(set(state.get("visited_video_ids", []))) < len(state.get("visited_video_ids", []))
        if state.get("visited_video_ids")
        else False,
        "cost": state.get("memory_runtime_stats", {}).get("cost", 0.0),
    }
    local_reward = compute_local_reward(previous_checker_state, checker_state, runtime_flags)

    feedback = f"Checker: {checker_state.reason}"
    if checker_state.missing_info and checker_state.signal == "planner":
        feedback += f" (Missing: {checker_state.missing_info})"

    experience_events = []
    if config.memory.enabled:
        memory_context = state.get("memory_context", {})
        retrieval_role = memory_context.get("retrieval_role", "checker")
        retrieval_query_text = memory_context.get("retrieval_query_text", user_query)
        retrieval_state_features = memory_context.get("retrieval_state_features")
        if not isinstance(retrieval_state_features, dict):
            retrieval_state_features = previous_checker_state.model_dump()
        event = ExperienceEvent(
            event_id=str(uuid.uuid4()),
            role=retrieval_role,
            query_text=retrieval_query_text,
            state_features=dict(retrieval_state_features),
            candidate_memory_ids=list(memory_context.get("candidate_memory_ids", [])),
            selected_memory_ids=list(memory_context.get("selected_memory_ids", [])),
            selector_scores=dict(memory_context.get("selector_scores", {})),
            checker_before=previous_checker_state.model_dump(),
            checker_after=checker_state.model_dump(),
            local_reward=local_reward,
            cost=runtime_flags,
            episode_id=state.get("episode_id"),
        )
        experience_events.append(event.model_dump())
        try:
            append_event(config.memory.events_path, event)
        except Exception as e:
            logger.log(
                "Checker",
                "event_logging_error",
                {"error": str(e), "events_path": config.memory.events_path},
                level="ERROR",
            )
        if getattr(getattr(config.memory, "bandit", None), "online_update", False):
            runtime = state.get("memory_runtime_stats", {}).get("runtime")
            if runtime is None:
                try:
                    runtime = get_default_memory_runtime(config)
                except Exception as e:
                    logger.log(
                        "Checker",
                        "bandit_runtime_unavailable",
                        {"error": str(e)},
                        level="WARN",
                    )
                    runtime = None
        else:
            runtime = None
        if runtime and getattr(runtime, "retriever", None) is not None and getattr(runtime.retriever, "selector", None) is not None:
            selector = runtime.retriever.selector
            is_neural_ucb = hasattr(selector, "emb_dim")
            selected_memory_ids = memory_context.get("selected_memory_ids", [])
            try:
                updated = False
                if is_neural_ucb:
                    query_emb = memory_context.get("query_embedding")
                    mem_embs = memory_context.get("selected_memory_embeddings") or {}
                    if query_emb is not None:
                        for memory_id in selected_memory_ids:
                            mem_emb = mem_embs.get(memory_id)
                            if mem_emb is None:
                                continue
                            selector.update(
                                np.asarray(query_emb, dtype=np.float32),
                                np.asarray(mem_emb, dtype=np.float32),
                                reward=local_reward,
                            )
                            updated = True
                    if updated:
                        bandit_cfg = config.memory.bandit
                        model_path = getattr(bandit_cfg, "model_path", None)
                        meta_path = getattr(bandit_cfg, "meta_path", None)
                        if model_path and meta_path:
                            selector.save(model_path, meta_path)
                else:
                    selected_feature_vectors = memory_context.get("selected_feature_vectors", {})
                    for memory_id in selected_memory_ids:
                        features = selected_feature_vectors.get(memory_id)
                        if features is None:
                            continue
                        selector.update(np.asarray(features, dtype=float), local_reward)
                        updated = True
                    if updated:
                        selector.save(config.memory.bandit_state_path)
            except Exception as e:
                logger.log(
                    "Checker",
                    "bandit_update_error",
                    {"error": str(e), "bandit_state_path": str(getattr(config.memory, "bandit_state_path", ""))},
                    level="ERROR",
                )

    return {
        "loop_step": current_step,
        "checker_state": checker_state.model_dump(),
        "previous_checker_state": previous_checker_state.model_dump(),
        "experience_events": experience_events,
        "routing_signal": checker_state.signal,
        "plan_trace": [feedback],
        "metrics": metrics,
        "step_log": [
            {
                "loop": state.get("loop_step", 0),
                "role": "checker",
                "answerable": checker_state.answerable,
                "drift": checker_state.drift,
                "missing_slots": list(checker_state.missing_slots),
                "verified_slots": list(checker_state.verified_slots),
                "signal": checker_state.signal,
                "reason": checker_state.reason,
                "confidence": checker_state.confidence,
            }
        ],
    }
