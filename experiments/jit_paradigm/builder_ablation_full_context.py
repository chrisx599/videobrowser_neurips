import json
import base64
import numpy as np
import io
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
    JIT Watcher (Ablation: Full Context - No Windowing):
    1. Selects top K videos using LLM based on metadata.
    2. Downloads video & transcript.
    3. SKIPS Window Identification. Just prepares the resource for the Analyst.
    """
    logger = get_logger()
    logger.log("JITWatcher", "start")
    print("🎥 [JIT Watcher - Full Context] Selecting videos and fetching data (No Windowing)...")
    config = get_config()
    metrics = state.get("metrics", {})
    video_store = state.get("video_store", {})
    
    raw_candidates = state.get("raw_candidates", [])
    if not raw_candidates:
        return {"video_store": video_store}

    # Use LLM to select Top K Candidates based on metadata
    print("   -> Selecting top videos using LLM...")
    llm = get_llm(node_name="watcher") 
    user_query = state.get("user_query", "")
    top_k = config.selector.top_k
    
    candidates_info = ""
    valid_candidates = []
    
    for i, raw in enumerate(raw_candidates):
        url = raw.get("link", "") or raw.get("videourl", "")
        if url:
            title = raw.get("title", "Unknown Title")
            desc = raw.get("snippet", "") or raw.get("description", "")
            candidates_info += f"[{i}] Title: {title}\n    Description: {desc}\n    URL: {url}\n\n"
            
            # Create a normalized candidate object
            candidate = raw.copy()
            candidate["url"] = url
            valid_candidates.append(candidate)
            
    if not valid_candidates:
        return {"video_store": video_store}

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
        
        if isinstance(selected_indices, list):
            for idx in selected_indices:
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
    
    for i, video in enumerate(target_videos):
        print(f"   -> Fetching data for Video {i+1}/{len(target_videos)}: {video.get('title', 'Unknown')}")
        video_url = video["url"]
        video_id = extract_youtube_id(video_url) or f"vid_{i}"
        
        # 1. Transcript with Timestamps
        transcript_segments = []
        transcript_text_with_timestamps = ""
        try:
            transcript_segments = fetch_transcript_with_timestamps(video_url)
            if transcript_segments:
                # Format: [00:00 - 00:05] Text...
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

        # Store in VideoResource - SKIPPING VLM WINDOWING
        # We just mark it as verified so the Analyst picks it up.
        resource = VideoResource(
            video_id=video_id,
            title=video.get('title', 'Unknown'),
            url=video_url,
            duration=video.get('duration', 'Unknown'),
            status="verified",
            summary="Full Context Ablation - No Window Analysis",
            transcript=transcript_text_with_timestamps
        )
        video_store[video_id] = resource

    return {
        "video_store": video_store,
        "metrics": metrics
    }

def jit_analyst_node(state: AgentState):
    """
    JIT Analyst (Ablation: Full Context):
    1. Consumes ALL verified videos.
    2. Extracts sparse frames (representing the whole video) + Full Transcript.
    3. Feeds everything to the LLM.
    """
    logger = get_logger()
    logger.log("JITAnalyst", "start")
    print("🧠 [JIT Analyst - Full Context] Analyzing full transcripts and sparse frames...")
    
    video_store = state.get("video_store", {})
    user_query = state.get("user_query", "")
    metrics = state.get("metrics", {})
    config = get_config()
    
    if not video_store:
        return {"final_answer": "No videos were successfully processed."}

    llm = get_llm(node_name="analyst")
    content_parts = []
    
    content_parts.append({
        "type": "text", 
        "text": f"User Query: {user_query}\n\nAnalyze the following video content (transcripts and sampled frames) to answer the query."
    })

    has_relevant_content = False

    for i, (vid, res) in enumerate(video_store.items()):
        print(f"   -> Processing Video {i+1}: {res.title}")
        
        # 1. Download Video to extract sparse frames
        frames_data = []
        try:
            video_path = download_video_file(res.url)
            if video_path:
                num_frames = config.analyst.num_frames
                frames_data = extract_frames_with_timestamps(video_path, num_frames=num_frames)
                print(f"      -> Extracted {len(frames_data)} sparse frames (Full Context).")
        except Exception as e:
            print(f"      ⚠️ Vision extraction error: {e}")
        
        if frames_data or res.transcript:
            has_relevant_content = True
            
            content_parts.append({
                "type": "text", 
                "text": f"\n=== Video: {res.title} ===\n"
            })
            
            # Add Frames
            if frames_data:
                for f in frames_data:
                    content_parts.append({
                        "type": "text",
                        "text": f"[Frame at {f['timestamp']:.1f}s]"
                    })
                    content_parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{f['image']}"}
                    })
            
            # Add Transcript
            if res.transcript:
                truncated_transcript = res.transcript[:25000] # Safety limit
                content_parts.append({
                    "type": "text",
                    "text": "Transcript:" + truncated_transcript
                })

    if not has_relevant_content:
        return {"final_answer": "No relevant video content or transcripts found to answer the query."}

    content_parts.append({
        "type": "text",
        "text": f"""\n        
        Based on the video frames and transcripts provided above, answer the User Query.
        Response Format:
        {{
            "Explanation": "your explanation for your final answer",
            "Answer": "your succinct, final answer",
            "Confidence": "your confidence score between 0% and 100% for your answer"
        }}
        """
    })

    print("   -> Invoking Analyst LLM with full context...")
    try:
        response = llm.invoke([HumanMessage(content=content_parts)])
        metrics = update_token_metrics(metrics, response, category="jit_analyst")
        final_answer = response.content
    except Exception as e:
        final_answer = f"Error: {e}"

    return {
        "final_answer": final_answer,
        "metrics": metrics
    }

def jit_checker_node(state: AgentState):
    """
    JIT Checker:
    Strictly controls the search rounds based on max_loop_steps.
    """
    logger = get_logger()
    config = get_config()
    max_loop_steps = config.checker.max_loop_steps
    current_step = state.get("loop_step", 0) + 1
    
    print(f"🧐 [JIT Checker] Step {current_step}/{max_loop_steps}")
    
    if current_step < max_loop_steps:
         print("   -> Round complete. Continuing search loop (strict mode)...")
         routing_signal = "planner"
    else:
         print("   -> Max loops reached. Proceeding to Analyst.")
         routing_signal = "analyst"

    return {
        "loop_step": current_step,
        "routing_signal": routing_signal
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
    load_config("experiments/jit_paradigm/config.yaml")
    
    app = build_graph()
    
    config = {"configurable": {"thread_id": "jit_demo_full_context"}}
    
    print("🚀 JIT Experiment (Full Context Baseline - No Windowing) Started...")
    
    inputs = {"user_query": "A legendary power forward, after switching careers to become a commentator, once bet with his co-host on a popular American basketball analysis show that a No. 1 draft pick center from Asia could not score 19 points in a single game. Subsequently, the center proved himself in a game, forcing the commentator to fulfill the bet — kissing a donkey's butt on a subsequent live broadcast. What was the center's final score in that game?"}
    
    for update in app.stream(inputs, config=config):
        for node_name, node_output in update.items():
            print(f"--- Step: {node_name} ---")
            if node_name == "analyst":
                print("\n✅ FINAL ANSWER:\n")
                print(node_output.get("final_answer", "No answer."))
                print("\n📊 Token Usage:\n")
                print(node_output.get("metrics", {}))
