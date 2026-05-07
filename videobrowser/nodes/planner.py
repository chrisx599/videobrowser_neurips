from langchain_core.messages import SystemMessage, HumanMessage
from videobrowser.core.state import AgentState, format_planner_view
from videobrowser.memory.bootstrap import get_default_memory_runtime
from videobrowser.memory.injection import render_memory_block
from videobrowser.memory.serializer import serialize_planner_state
from videobrowser.utils.prompt_manager import load_prompt
from videobrowser.utils.parser import extract_json_from_text
from videobrowser.utils.metrics import update_token_metrics
from videobrowser.utils.logger import get_logger
from videobrowser.config import get_config
from dotenv import load_dotenv
from videobrowser.utils.llm_factory import get_llm
import json

load_dotenv()

llm = get_llm(node_name="planner")
logger = get_logger()

def planner_node(state: AgentState):
    logger.log("Planner", "start", {"loop_step": state.get("loop_step", 0)})
    config = get_config()

    context_view = format_planner_view(state)
    retrieval_context = None
    memory_context = {
        "candidate_memory_ids": [],
        "selected_memory_ids": [],
        "selector_scores": {},
    }
    memory_block = ""
    runtime = state.get("memory_runtime_stats", {}).get("runtime") or get_default_memory_runtime(config)
    if (
        config.memory.enabled
        and runtime
        and getattr(runtime, "retriever", None)
    ):
        retrieval_context = serialize_planner_state(state)
        retrieval_context.proposal_top_n = config.memory.retrieval.proposal_top_n
        retrieval_context.selection_top_k = config.memory.retrieval.selection_top_k
        retrieval_context.filter_tags["include_failure_memories"] = (
            config.memory.retrieval.include_failure_memories
        )
        retrieval = runtime.retriever.retrieve_with_meta(retrieval_context)

        action = retrieval.recommended_action or "apply"
        drop_threshold = float(getattr(config.memory.critic, "per_card_drop_threshold", 0.0))
        if action == "skip":
            memory_block = ""
        else:
            memory_block = render_memory_block(
                "planner",
                retrieval.selected_cards,
                applicability_scores=retrieval.applicability_scores,
                drop_threshold=drop_threshold,
                low_confidence_nudge=(action == "loop"),
            )

        memory_context = {
            "retrieval_role": retrieval_context.role,
            "retrieval_query_text": retrieval_context.query_text,
            "retrieval_state_features": retrieval_context.state_features,
            "candidate_memory_ids": [card.memory_id for card in retrieval.candidate_cards],
            "selected_memory_ids": [card.memory_id for card in retrieval.selected_cards],
            "selector_scores": retrieval.selector_scores,
            "selected_feature_vectors": retrieval.selected_feature_vectors,
            "query_embedding": getattr(retrieval, "query_embedding", None).tolist() if getattr(retrieval, "query_embedding", None) is not None else None,
            "selected_memory_embeddings": {k: v.tolist() for k, v in getattr(retrieval, "selected_memory_embeddings", {}).items()} if getattr(retrieval, "selected_memory_embeddings", None) else None,
            "applicability_scores": retrieval.applicability_scores,
            "critic_confidence": retrieval.critic_confidence,
            "critic_rationale": retrieval.critic_rationale,
            "retrieval_uncertainty": retrieval.retrieval_uncertainty,
            "uncertainty_components": retrieval.uncertainty_components,
            "combined_confidence": retrieval.combined_confidence,
            "recommended_action": action,
            "ranker_source": retrieval.ranker_source,
        }

    system_prompt = load_prompt(
        "planner.j2",
        max_queries=config.planner.max_queries,
        memory_block=memory_block,
    )
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=context_view)
    ]
    response = llm.invoke(messages)
    
    metrics = update_token_metrics(state.get("metrics", {}), response)
    
    try:
        plan = extract_json_from_text(response.content)
        # Enforce max_queries limit
        if plan.get("search_queries"):
             plan["search_queries"] = plan["search_queries"][:config.planner.max_queries]
    except Exception as e:
        print(f"⚠️ [Planner] JSON parsing failed: {e}. Fallback to user query.")
        logger.log("Planner", "error", {"error": str(e), "content": response.content}, level="ERROR")
        plan = {
            "thought": "Parsing error, fallback to original query.",
            "search_queries": [state["user_query"]]
        }

    if not plan.get("search_queries"):
        print("🧠 [Planner] No new queries — signalling loop termination.")

    print(f"🧠 [Planner] Thought: {plan.get('thought')}")
    print(f"🔍 [Planner] Queries: {plan.get('search_queries')}")

    logger.log("Planner", "end", {
        "thought": plan.get("thought"),
        "search_queries": plan.get("search_queries")
    })

    return {
        "plan_trace": [f"Thought: {plan.get('thought')}"],
        "current_search_queries": plan.get("search_queries", []),
        "metrics": metrics,
        "memory_context": memory_context,
        "step_log": [
            {
                "loop": state.get("loop_step", 0),
                "role": "planner",
                "thought": plan.get("thought"),
                "search_queries": plan.get("search_queries", []),
                "tokens_total_so_far": metrics.get("total_tokens", 0),
                "memory_used": bool(memory_block),
            }
        ],
    }
