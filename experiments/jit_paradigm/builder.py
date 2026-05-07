import json
import base64
import numpy as np
import io
import os
from PIL import Image
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, SystemMessage

from videobrowser.core.state import AgentState, VideoResource
from videobrowser.utils.parser import extract_youtube_id, extract_json_from_text
from videobrowser.utils.llm_factory import get_llm
from videobrowser.tools.fetch_video import fetch_transcript_with_timestamps, download_video_file
from videobrowser.tools.vision import extract_frames_with_timestamps, extract_frames_from_window
from videobrowser.config import load_config, get_config
from videobrowser.memory.bootstrap import build_memory_runtime
from videobrowser.utils.prompt_manager import load_prompt
from videobrowser.utils.metrics import update_token_metrics
from videobrowser.utils.logger import get_logger

from videobrowser.nodes.planner import planner_node
from videobrowser.nodes.searcher import searcher_node

try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None


def jit_watcher_node(state: AgentState):
    """
    JIT Watcher:
    1. Selects top K videos using LLM based on metadata.
    2. For each video:
       - Downloads video & transcript (with timestamps).
       - Extracts sparse frames (e.g. 16) with timestamps.
       - Uses VLM to identify the RELEVANT TEMPORAL WINDOWS.
    3. Stores the window info in video_store.
    """
    logger = get_logger()
    logger.log("JITWatcher", "start")
    print("🎥 [JIT Watcher] Starting sparse sampling and window identification...")
    config = get_config()
    metrics = state.get("metrics", {})
    video_store = state.get("video_store", {})
    
    raw_candidates = state.get("raw_candidates", [])
    if not raw_candidates:
        return {
            "video_store": video_store,
            "step_log": [
                {
                    "loop": state.get("loop_step", 0),
                    "role": "watcher",
                    "videos_processed": [],
                }
            ],
        }

    # Use LLM to select Top K Candidates based on metadata
    print("   -> Selecting top videos using LLM...")
    llm = get_llm(node_name="watcher")
    user_query = state.get("user_query", "")
    # VideoBrowsecomp questions are intentionally indirect ("the actor who
    # starred in 'John Wick'..." actually points at *Good Fortune*). The
    # planner's search queries are the concrete terms that pulled this
    # candidate set out of BM25; threading them through to Stage 2 gives the
    # window-finder VLM a non-cryptic anchor for what the user is really
    # asking about. We dedupe (preserving order) and prefer the most recent
    # planner queries when current_search_queries is empty.
    _seen_q: set[str] = set()
    planner_search_queries: list[str] = []
    for _q in (state.get("current_search_queries") or []) + (state.get("tried_queries") or []):
        _q = (_q or "").strip()
        if not _q or _q in _seen_q:
            continue
        _seen_q.add(_q)
        planner_search_queries.append(_q)

    # Checker's diagnosis from the previous loop. Empty {} on loop 0 (checker
    # hasn't run yet) — handled downstream by rendering an empty prompt block.
    checker_state_dict = state.get("checker_state", {}) or {}
    checker_missing_slots = list(checker_state_dict.get("missing_slots") or [])
    checker_verified_slots = list(checker_state_dict.get("verified_slots") or [])
    checker_missing_info = (checker_state_dict.get("missing_info") or "").strip()

    top_k = config.selector.top_k

    candidates_info = ""
    valid_candidates = []
    skipped_seen = 0

    for raw in raw_candidates:
        url = raw.get("link", "") or raw.get("videourl", "")
        if not url:
            continue
        vid = extract_youtube_id(url)
        if vid and vid in video_store:
            skipped_seen += 1
            continue
        idx = len(valid_candidates)
        title = raw.get("title", "Unknown Title")
        desc = raw.get("snippet", "") or raw.get("description", "")
        candidates_info += f"[{idx}] Title: {title}\n    Description: {desc}\n    URL: {url}\n\n"

        candidate = raw.copy()
        candidate["url"] = url
        valid_candidates.append(candidate)

    if skipped_seen:
        print(f"   -> Pre-filtered {skipped_seen} already-seen candidates.")
    if not valid_candidates:
        print("   -> No new candidates after filtering. Returning to checker.")
        return {
            "video_store": video_store,
            "step_log": [
                {
                    "loop": state.get("loop_step", 0),
                    "role": "watcher",
                    "videos_processed": [],
                }
            ],
        }

    selection_prompt = f"""
    User Query: "{user_query}"
    
    You are provided with a list of video search results. 
    Select the top {top_k} most relevant videos that are likely to contain the answer to the User Query.
    
    Candidates:
    {candidates_info}
    
    Return the indices of the selected videos as a JSON list of integers.
    Example: [0, 2, 4]
    """

    target_videos = []
    try:
        response = llm.invoke([HumanMessage(content=selection_prompt)])
        metrics = update_token_metrics(metrics, response, category="jit_selector")
        selected_indices = extract_json_from_text(response.content)

        print(f"   -> LLM selected video indices: {selected_indices}")
        
        if isinstance(selected_indices, list):
            for idx in selected_indices:
                if isinstance(idx, int) and 0 <= idx < len(valid_candidates):
                    target_videos.append(valid_candidates[idx])
                    if len(target_videos) >= top_k:
                        break
        elif isinstance(selected_indices, dict):
            # Handle case where LLM returns a dict, directly process value
            values = selected_indices.values()
            for idx in values:
                if isinstance(idx, int) and 0 <= idx < len(valid_candidates):
                    target_videos.append(valid_candidates[idx])
                    if len(target_videos) >= top_k:
                        break
        else:
             print(f"      ⚠️ Invalid selection format, falling back to top {top_k}.")
             target_videos = valid_candidates[:top_k]
             
    except Exception as e:
        print(f"      ⚠️ Selection error: {e}, falling back to top {top_k}.")
        target_videos = valid_candidates[:top_k]

    if not target_videos:
         target_videos = valid_candidates[:top_k]

    print(f"   -> Selected {len(target_videos)} videos for processing.")

    videos_log_for_step = []

    for i, video in enumerate(target_videos):
        video_url = video["url"]
        video_id = extract_youtube_id(video_url) or f"vid_{i}"

        if video_id in video_store:
            print(f"   -> Skipping Video {i+1}/{len(target_videos)} (already in store): {video.get('title', 'Unknown')}")
            continue

        print(f"   -> Processing Video {i+1}/{len(target_videos)}: {video.get('title', 'Unknown')}")

        # 1. Download & Vision (Sparse Sampling)
        frames_data = []
        video_path = None
        try:
            video_path = download_video_file(video_url)
            if video_path:
                num_frames = config.watcher.num_frames  # e.g., 16
                frames_data = extract_frames_with_timestamps(video_path, num_frames=num_frames)
                print(f"      -> Extracted {len(frames_data)} frames with timestamps.")
        except Exception as e:
            print(f"      ⚠️ Vision error: {e}")

        # 2. Transcript with Timestamps
        transcript_segments = []
        transcript_text_with_timestamps = ""
        try:
            transcript_segments = fetch_transcript_with_timestamps(video_url)
            if transcript_segments:
                # Track Whisper Usage (JIT specific approximation using video duration)
                if config.transcript.provider == "whisper" and VideoReader and video_path and os.path.exists(video_path):
                     try:
                         vr = VideoReader(video_path, ctx=cpu(0))
                         # Calculate duration in seconds
                         duration = len(vr) / vr.get_avg_fps()
                         metrics["whisper_audio_seconds"] = metrics.get("whisper_audio_seconds", 0.0) + duration
                     except Exception as e:
                         print(f"      ⚠️ Could not calculate duration for Whisper metric: {e}")
                
                # Format for VLM prompt: [00:00 - 00:05] Text...
                lines = []
                for seg in transcript_segments:
                    start = seg.get('start', 0)
                    end = seg.get('end', 0)
                    text = seg.get('text', '')
                    lines.append(f"[{start:.1f}s - {end:.1f}s] {text}")
                
                transcript_text_with_timestamps = "\n".join(lines)
                print(f"      -> Fetched transcript ({len(transcript_segments)} segments).")
            else:
                print("      -> No transcript found.")
        except Exception as e:
            print(f"      ⚠️ Transcript error: {e}")

        # 3. Identify Temporal Window using VLM
        # Prepare frame descriptions
        frame_descriptions = "\n".join([f"Frame {idx+1}: Timestamp {f['timestamp']:.2f}s" for idx, f in enumerate(frames_data)])
        
        # Truncate transcript if too long (approx 20k chars)
        truncated_transcript = transcript_text_with_timestamps[:25000]
        
        # Build the planner-search-queries block. Empty list (e.g. very first
        # loop before planner has spoken) safely renders an empty section.
        if planner_search_queries:
            _q_lines = "\n".join(f"          - \"{q}\"" for q in planner_search_queries)
            search_query_block = (
                "Planner's concrete search queries that surfaced this video:\n"
                f"{_q_lines}\n"
                "These terms are what the User Query is *actually* asking about. "
                "User Queries in this benchmark are intentionally indirect — they "
                "describe the answer obliquely, often via cues to other works or people "
                "(e.g. \"the actor from movie X\" when the target video is *not* movie X). "
                "Trust the search queries above as the literal subject of the search."
            )
        else:
            search_query_block = ""

        # Checker gap block — turns the watcher from open-ended evidence hunt
        # into targeted gap-fill when the checker has already diagnosed what's
        # missing. Empty on loop 0 (no prior checker run).
        if checker_missing_slots or checker_verified_slots or checker_missing_info:
            _gap_lines = []
            if checker_missing_slots:
                _gap_lines.append(
                    "          - Still missing: " + ", ".join(checker_missing_slots)
                )
            if checker_missing_info:
                _gap_lines.append(
                    f"          - Why we re-searched: {checker_missing_info}"
                )
            if checker_verified_slots:
                _gap_lines.append(
                    "          - Already verified (no need to re-prove): "
                    + ", ".join(checker_verified_slots)
                )
            checker_gap_block = (
                "Checker's diagnosis from the previous loop:\n"
                + "\n".join(_gap_lines)
                + "\nPrioritize windows whose visual or transcript content could "
                "fill the missing items above. Treat the verified items as settled — "
                "do not spend windows re-confirming them."
            )
        else:
            checker_gap_block = ""

        prompt_text = f"""
        You are a video investigator.
        User Query: "{user_query}"

        {search_query_block}

        {checker_gap_block}

        Video Title: {video.get('title', 'Unknown')}

        I have sampled {len(frames_data)} frames from the video at specific timestamps:
        {frame_descriptions}

        Transcript (with timestamps):
        {truncated_transcript}

        Task:
        Identify all specific temporal windows (start_time to end_time) in this video that could plausibly contain evidence for answering the User Query.
        - Combine visual cues from the frames and semantic cues from the transcript.
        - If the answer is in the transcript, use its timestamps.
        - If the answer is visual, use the frame timestamps to estimate the window.
        - When in doubt, RETURN A WINDOW. Stage 3 will re-extract dense frames inside it; an empty window list means no further inspection happens. Err on the side of inclusion.
        - Mark relevant=false ONLY if NO visible scene AND NO transcript segment plausibly relates to ANY of the planner's search queries above (or to the User Query when no planner queries were provided).
        - Provide a brief reasoning for each selected window.

        Output Format (JSON):
        {{
            "relevant": true/false,
            "windows": [
                {{
                    "start_time_seconds": <float>,
                    "end_time_seconds": <float>,
                    "reasoning": "..."
                }},
                ...
            ]
        }}
        """
        
        content_parts = [{"type": "text", "text": prompt_text}]
        for f in frames_data:
            content_parts.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{f['image']}"}
            })
            
        print(f"      -> Identifying relevant window with VLM...")
        analysis_result = {}
        summary_text = "Analysis failed."
        
        try:
            response = llm.invoke([HumanMessage(content=content_parts)])
            metrics = update_token_metrics(metrics, response, category="jit_watcher")
            raw_response = response.content
            
            # Robust JSON extraction
            try:
                analysis_result = extract_json_from_text(raw_response)
                summary_text = json.dumps(analysis_result, indent=2)
            except Exception as e:
                print(f"      ⚠️ JSON parsing error: {e}")
                summary_text = raw_response

            print(f"      -> Window identified: {summary_text[:100]}...")
            
        except Exception as e:
            print(f"      ⚠️ LLM error: {e}")
            summary_text = f"Error: {e}"

        # Store in VideoResource
        resource = VideoResource(
            video_id=video_id,
            title=video.get('title', 'Unknown'),
            url=video_url,
            duration=video.get('duration', 'Unknown'),
            status="verified",
            summary=summary_text,
            transcript=transcript_text_with_timestamps
        )
        video_store[video_id] = resource

        # Collect per-video log entry
        _relevant = bool(analysis_result.get("relevant", False)) if isinstance(analysis_result, dict) else False
        _windows = analysis_result.get("windows", []) if isinstance(analysis_result, dict) else []
        _frame_indices = [f.get("timestamp") for f in frames_data]
        videos_log_for_step.append({
            "video_id": video_id,
            "title": video.get("title", "Unknown"),
            "stage1_selected": True,
            "stage2_frames_picked": _frame_indices,
            "stage3_zoomed_ranges": [],
            "relevant": _relevant,
            "n_windows": len(_windows),
            "final_status": "verified",
        })

    return {
        "video_store": video_store,
        "metrics": metrics,
        "step_log": [
            {
                "loop": state.get("loop_step", 0),
                "role": "watcher",
                "videos_processed": videos_log_for_step,
            }
        ],
    }

def jit_analyst_node(state: AgentState):
    """
    JIT Analyst:
    1. Reads the identified temporal windows from video_store.
    2. Extracts 1 FPS frames for those specific windows.
    3. Feeds relevant frames and transcript segments to LLM for final answer.
    """
    logger = get_logger()
    logger.log("JITAnalyst", "start")
    print("🧠 [JIT Analyst] Extracting relevant clips and synthesizing final answer...")
    
    video_store = state.get("video_store", {})
    user_query = state.get("user_query", "")
    metrics = state.get("metrics", {})
    config = get_config()
    prompts_config = getattr(config, "prompts", None)
    format_instructions = getattr(prompts_config, "analyst_format_instructions", "") or ""
    
    if not video_store:
        return {
            "final_answer": "No videos were successfully processed.",
            "step_log": [
                {
                    "loop": state.get("loop_step", 0),
                    "role": "analyst",
                    "final_answer": "No videos were successfully processed.",
                    "has_evidence": False,
                    "evidence_summary": "",
                }
            ],
        }

    llm = get_llm(node_name="analyst")
    content_parts = []

    prompt_text = load_prompt(
        "analyst_report.j2",
        user_query=user_query,
        memory_block="",
        format_instructions=format_instructions,
    )
    content_parts.append({"type": "text", "text": prompt_text})

    has_relevant_content = False

    for i, (vid, res) in enumerate(video_store.items()):
        # Parse the window analysis JSON
        try:
            analysis = json.loads(res.summary)
        except:
            print(f"   -> Skipping Video {i+1}: Could not parse window analysis.")
            continue
            
        if not analysis.get("relevant", False):
            continue
            
        windows = analysis.get("windows", [])
        if not windows:
            # Fallback for old format if 'windows' key missing but relevant is true
            if "start_time_seconds" in analysis:
                windows = [analysis]
            else:
                continue
        
        print(f"   -> Processing Video {i+1}: {res.title} ({len(windows)} windows)")
        
        # We re-call download_video_file which hits cache.
        video_path = download_video_file(res.url)
        if not video_path:
            print(f"      ⚠️ Could not retrieve video file for {res.title}")
            continue

        full_transcript_lines = res.transcript.splitlines() 
        
        for win in windows:
            start = win.get("start_time_seconds", 0.0)
            end = win.get("end_time_seconds", 0.0)
            reason = win.get("reasoning", "")
            
            if end <= start:
                continue

            has_relevant_content = True
            
            # 1. Extract Frames at 1 FPS
            print(f"      -> Extracting clip {start:.1f}s - {end:.1f}s (1 FPS)...")
            
            # Use shared vision tool
            frames = extract_frames_from_window(video_path, start, end, fps_sample=1.0)
            
            if frames:
                content_parts.append({
                    "type": "text", 
                    "text": f"\n=== Video: {res.title} [Clip: {start:.1f}s - {end:.1f}s] ===\nReasoning: {reason}\n"
                })
                
                for f in frames:
                    content_parts.append({
                        "type": "text",
                        "text": f"[Frame at {f['timestamp']:.1f}s]"
                    })
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{f['image']}"}
                    })

            # 2. Extract Relevant Transcript
            relevant_lines = []
            for line in full_transcript_lines:
                try:
                    import re
                    match = re.search(r"((\d+\.?\d*)s\s*-\s*(\d+\.?\d*)s)", line)
                    if match:
                        t_start = float(match.group(2))
                        t_end = float(match.group(3))
                        
                        # Check overlap
                        if t_end >= start and t_start <= end:
                            relevant_lines.append(line)
                except:
                    pass
            
            if relevant_lines:
                content_parts.append({
                    "type": "text",
                    "text": "Transcript Segment:\n" + "\n".join(relevant_lines)
                })
                print(f"      -> Added {relevant_lines} transcript lines.")

    if not has_relevant_content:
        print("   -> No specific visual windows identified. Falling back to full transcript analysis...")
        
        # Fallback: Use full transcripts if available
        for i, (vid, res) in enumerate(video_store.items()):
            if res.transcript:
                print(f"      -> Adding transcript for Video {i+1}: {res.title}")
                content_parts.append({
                    "type": "text", 
                    "text": f"\n=== Video Transcript: {res.title} ===\n{res.transcript[:25000]}..." # Limit to avoid context overflow if huge
                })
                has_relevant_content = True
        
        if not has_relevant_content:
            return {
                "final_answer": "No relevant video content or transcripts found to answer the query.",
                "step_log": [
                    {
                        "loop": state.get("loop_step", 0),
                        "role": "analyst",
                        "final_answer": "No relevant video content or transcripts found to answer the query.",
                        "has_evidence": False,
                        "evidence_summary": "",
                    }
                ],
            }

    content_parts.append(
        {
            "type": "text",
            "text": "\nAnswer the user query using only the evidence above.",
        }
    )

    print("   -> Invoking Analyst LLM with video context...")
    try:
        response = llm.invoke([HumanMessage(content=content_parts)])
        metrics = update_token_metrics(metrics, response, category="jit_analyst")
        final_answer = response.content
    except Exception as e:
        final_answer = f"Error: {e}"

    return {
        "final_answer": final_answer,
        "metrics": metrics,
        "step_log": [
            {
                "loop": state.get("loop_step", 0),
                "role": "analyst",
                "final_answer": final_answer,
                "has_evidence": has_relevant_content,
                "evidence_summary": "",
            }
        ],
    }

def jit_checker_node(state: AgentState):
    """
    JIT Checker:
    Controls search rounds. Hard ceiling is config.checker.max_loop_steps.
    """
    logger = get_logger()
    config = get_config()
    max_loop_steps = config.checker.max_loop_steps
    current_step = state.get("loop_step", 0) + 1

    video_store_size = len(state.get("video_store", {}))
    history_size = len(state.get("tried_queries", []))

    print(f"🧐 [JIT Checker] Step {current_step}/{max_loop_steps}")
    print(f"   -> Knowledge Accumulation: {video_store_size} videos in store, {history_size} queries tried.")

    last_queries = state.get("current_search_queries", [])

    if not last_queries and video_store_size > 0:
        print("   -> Planner signalled done (empty queries). Proceeding to Analyst.")
        routing_signal = "analyst"
    elif current_step < max_loop_steps:
        print("   -> Round complete. Continuing search loop (strict mode)...")
        routing_signal = "planner"
    else:
        print("   -> Max loops reached. Proceeding to Analyst.")
        routing_signal = "analyst"

    return {
        "loop_step": current_step,
        "routing_signal": routing_signal,
        "step_log": [
            {
                "loop": state.get("loop_step", 0),
                "role": "checker",
                "answerable": routing_signal == "analyst",
                "missing_slots": [],
                "verified_slots": [],
                "signal": routing_signal,
                "reason": f"Step {current_step}/{max_loop_steps}",
                "confidence": None,
            }
        ],
    }

def route_jit_checker(state: AgentState):
    return state.get("routing_signal", "analyst")

def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("searcher", searcher_node)
    workflow.add_node("watcher", jit_watcher_node)
    workflow.add_node("checker", jit_checker_node)
    workflow.add_node("analyst", jit_analyst_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "searcher")
    workflow.add_edge("searcher", "watcher")
    workflow.add_edge("watcher", "checker")
    
    workflow.add_conditional_edges(
        "checker",
        route_jit_checker,
        {
            "planner": "planner",
            "analyst": "analyst"
        }
    )
    
    workflow.add_edge("analyst", END)
    
    memory = MemorySaver()
    
    return workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    config_obj = load_config("experiments/jit_paradigm/config.yaml")
    memory_runtime = build_memory_runtime(config_obj)
    
    app = build_graph()
    
    config = {"configurable": {"thread_id": "jit_demo"}}
    
    print("🚀 JIT Experiment (Sparse Sampling + Window Finding) Started...")
    
    inputs = {
        "user_query": "A legendary power forward, after switching careers to become a commentator, once bet with his co-host on a popular American basketball analysis show that a No. 1 draft pick center from Asia could not score 19 points in a single game. Subsequently, the center proved himself in a game, forcing the commentator to fulfill the bet — kissing a donkey's butt on a subsequent live broadcast. What was the center's final score in that game?",
        "memory_runtime_stats": {"runtime": memory_runtime},
    }
    
    for update in app.stream(inputs, config=config):
        for node_name, node_output in update.items():
            print(f"--- Step: {node_name} ---")
            if node_name == "analyst":
                print("\n✅ FINAL ANSWER:\n")
                print(node_output.get("final_answer", "No answer."))
                print("\n📊 Token Usage:\n")
                print(node_output.get("metrics", {}))
