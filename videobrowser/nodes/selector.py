from videobrowser.core.state import AgentState, VideoResource
from videobrowser.utils.prompt_manager import load_prompt
from videobrowser.utils.parser import extract_json_from_text
from videobrowser.utils.metrics import update_token_metrics
from videobrowser.utils.logger import get_logger
from langchain_core.messages import SystemMessage, HumanMessage
from dotenv import load_dotenv
from videobrowser.utils.llm_factory import get_llm
from videobrowser.config import get_config
import json

load_dotenv()

llm = get_llm(node_name="selector")
config = get_config()
logger = get_logger()

def selector_node(state: AgentState):
    # --- 1. Determine Running Mode ---
    
    # Mode A: Pre-Selection
    # If raw_candidates exist, it means search just finished.
    if state.get("raw_candidates") and len(state["raw_candidates"]) > 0:
        return _run_pre_selection(state)
    
    # Mode B: Post-Verification
    # Check if there are 'watched' videos in the Video Store.
    # 'watched' means the Watcher just finished processing them, but Selector hasn't verified them yet.
    watched_videos = [
        v for v in state["video_store"].values() 
        if v.status == "watched"
    ]
    if watched_videos:
        return _run_post_verification(state, watched_videos)
        
    # If no search results and no videos to verify, possibly search returned empty.
    # In this case, Selector does nothing, letting the Graph routing logic handle it 
    # (usually goes to Checker, which might send it back to Planner if no progress).
    logger.log("Selector", "idle", {"reason": "no_candidates_or_watched"})
    return {}

# --- Mode A: Pre-Selection ---
def _run_pre_selection(state: AgentState):
    print("🗑️ [Selector] Mode A: Initial filtering...")
    raw_list = state["raw_candidates"]
    query = state["user_query"]
    
    logger.log("Selector", "start_filter", {"candidate_count": len(raw_list)})
    
    top_k = config.selector.top_k
    
    # 1. Construct Prompt
    prompt_text = load_prompt(
        "selector_filter.j2", 
        user_query=query, 
        candidates=raw_list,
        top_k=top_k
    )
    
    # 2. Call LLM
    metrics = state.get("metrics", {})
    try:
        response = llm.invoke([HumanMessage(content=prompt_text)])
        metrics = update_token_metrics(metrics, response)
        result = extract_json_from_text(response.content)
        selected_indices = result.get("selected_indices", [])
        reason = result.get("reason", "")
        print(f"   -> Selected {len(selected_indices)} videos. Reason: {reason}")
        
        logger.log("Selector", "filtered", {
            "selected_count": len(selected_indices),
            "reason": reason,
            "indices": selected_indices
        })
    except Exception as e:
        print(f"⚠️ [Selector] Mode A failed: {e}")
        logger.log("Selector", "error", {"mode": "A", "error": str(e)}, level="ERROR")
        selected_indices = []

    # 3. Update State
    new_videos = {}
    for idx in selected_indices:
        if 0 <= idx < len(raw_list):
            vid_data = raw_list[idx]
            # Use Link as ID to prevent duplicates
            # Note: raw_candidates dict already has a temp 'id' (usually link)
            vid_id = vid_data.get('id') or vid_data.get('link')
            
            # If video is already in store (e.g., passed before), skip to avoid re-watching
            if vid_id in state.get("video_store", {}):
                continue

            v_res = VideoResource(
                video_id=vid_id,
                title=vid_data.get('title', 'Unknown Title'),
                url=vid_data.get('link', ''),
                duration=vid_data.get('duration', 'unknown'),
                status="candidate",  # Set to candidate, waiting for Watcher to pick (or directly analyzing)
                relevance_reason=result.get("reason", "Selected by AI"),
                summary=vid_data.get('snippet', '') # Temporarily store snippet as summary
            )
            new_videos[v_res.video_id] = v_res
    
    return {
        "video_store": {**state.get("video_store", {}), **new_videos},
        "raw_candidates": [], # Clear raw_candidates, indicating pre-selection is done
        "metrics": metrics
    }

# --- Mode B: Post-Verification ---
def _run_post_verification(state: AgentState, videos_to_verify: list):
    print(f"⚖️ [Selector] Mode B: Verifying evidence for {len(videos_to_verify)} videos...")
    logger.log("Selector", "start_verification", {"count": len(videos_to_verify)})
    
    updated_videos = {}
    metrics = state.get("metrics", {})
    
    for video in videos_to_verify:
        # 1. Construct Prompt
        # Convert evidence list to readable text
        evidence_str = "\n".join([f"- [{e.source}] {e.content}" for e in video.evidence])
        
        prompt_text = load_prompt(
            "selector_verify.j2",
            user_query=state['user_query'],
            video_title=video.title,
            video_url=video.url,
            evidence=evidence_str
        )

        # 2. Call LLM
        try:
            response = llm.invoke([HumanMessage(content=prompt_text)])
            metrics = update_token_metrics(metrics, response)
            result = extract_json_from_text(response.content)
            new_status = result.get("status", "rejected")
            reason = result.get("reason", "")
            
            logger.log("Selector", "verified_video", {
                "video_id": video.video_id,
                "status": new_status,
                "reason": reason
            })
        except Exception as e:
            print(f"⚠️ [Selector] Verification failed for {video.video_id}: {e}")
            logger.log("Selector", "error", {"mode": "B", "video_id": video.video_id, "error": str(e)}, level="ERROR")
            new_status = "rejected"
            reason = "Verification error"

        # 3. Update Video Object
        # Note: Must update status here, otherwise infinite loop
        video.status = new_status 
        video.relevance_reason = reason # Update reason with verification feedback
        
        updated_videos[video.video_id] = video
        print(f"   -> Video '{video.title[:20]}...' is {new_status.upper()}. ({reason})")

    # Return updated video_store
    # Note: We only update changed videos, but StateGraph reducer (if default dict update)
    # needs to ensure we don't overwrite other videos. VideoStore is a Dict.
    # If we return {"video_store": updated_videos},
    # in LangGraph, if reducer is update operation (dict.update), then these keys will be updated, others kept.
    # Check state.py -> It is MessagesState, default overwrites key.
    # So we must return the full video_store or rely on custom reducer.
    # For safety, we manually merge.
    
    full_store = {**state["video_store"], **updated_videos}
    
    return {
        "video_store": full_store,
        "metrics": metrics
    }
