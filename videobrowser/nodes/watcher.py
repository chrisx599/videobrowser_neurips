import base64
from videobrowser.core.state import AgentState, EvidenceFragment
from videobrowser.tools.fetch_video import fetch_youtube_video_transcript, download_video_file
from videobrowser.tools.vision import extract_frames_from_video
from videobrowser.utils.prompt_manager import load_prompt
from videobrowser.utils.parser import extract_json_from_text
from videobrowser.utils.metrics import update_token_metrics
from videobrowser.utils.logger import get_logger
from videobrowser.utils.cache import cache_manager
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from videobrowser.utils.llm_factory import get_llm
import json
import os

load_dotenv()

llm = get_llm(node_name="watcher")
logger = get_logger()

def watcher_node(state: AgentState):
    print("👀 [Watcher] Scanning for candidate videos...")
    
    # 1. Identify Candidates
    candidates = [
        v for v in state["video_store"].values() 
        if v.status == "candidate"
    ]
    
    if not candidates:
        print("   -> No candidates to watch.")
        logger.log("Watcher", "idle", {"reason": "no_candidates"})
        return {
            "step_log": [
                {
                    "loop": state.get("loop_step", 0),
                    "role": "watcher",
                    "videos_processed": [],
                }
            ]
        }
        
    updated_videos = {}
    metrics = state.get("metrics", {})
    videos_log = []

    # Get config for vision settings
    config = cache_manager.refresh().config
    num_frames = config.watcher.num_frames if hasattr(config, "watcher") else 10

    for video in candidates:
        print(f"   -> Watching: {video.title[:30]}...")
        logger.log("Watcher", "start_watch", {"video_id": video.video_id, "title": video.title})
        
        # 2. Vision Processing (Download Video First)
        vision_result = None
        # download_video_file handles caching automatically if config.cache.enabled is True
        # We download FIRST so that transcript fetching (if using Whisper) can use the cached file.
        video_path = download_video_file(video.url)
        
        if video_path and os.path.exists(video_path):
            print(f"      🎥 Video file available, extracting frames...")
            frames = extract_frames_from_video(video_path, num_frames=num_frames)
            
            if frames:
                try:
                    # Prepare message with images
                    prompt_text = load_prompt(
                        "watcher_vision.j2",
                        user_query=state["user_query"],
                        video_title=video.title
                    )
                    
                    content_parts = [{"type": "text", "text": prompt_text}]
                    for frame_b64 in frames:
                        content_parts.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{frame_b64}"}
                        })
                        
                    # Invoke VLM
                    response = llm.invoke([HumanMessage(content=content_parts)])
                    metrics = update_token_metrics(metrics, response, category="watcher")
                    
                    vision_data = extract_json_from_text(response.content)
                    
                    # Cache Caption
                    caption = vision_data.get("caption")
                    if caption:
                        cache_manager.save_caption(video.video_id, caption)
                        print(f"      📝 Caption cached.")
                        
                    vision_result = vision_data
                    
                except Exception as e:
                    print(f"      ❌ Vision analysis failed: {e}")
                    logger.log("Watcher", "vision_error", {"video_id": video.video_id, "error": str(e)}, level="ERROR")

        # 3. Fetch Content (Transcript)
        full_transcript = ""
        try:
            docs = fetch_youtube_video_transcript(video.url)
            if docs:
                # Track Whisper Usage
                for d in docs:
                    if d.metadata.get("provider") == "whisper":
                        dur = d.metadata.get("audio_duration", 0.0)
                        if dur > 0:
                            metrics["whisper_audio_seconds"] = metrics.get("whisper_audio_seconds", 0.0) + dur

                full_transcript = "\n".join([d.page_content for d in docs])
                full_transcript = full_transcript[:15000] # Truncate
            else:
                print(f"      ⚠️ No transcript found for {video.video_id}")
        except Exception as e:
            print(f"      ⚠️ Transcript fetch warning: {e}")
        
        # 4. Synthesize Evidence (Transcript + Vision)
        # If we have vision result, use it. Otherwise fall back to transcript extraction
        
        evidence_list = []
        
        if vision_result:
            # Process Vision Result
            # Add vision fragments
            for f in vision_result.get("fragments", []):
                 ts_start = f.get("timestamp_start")
                 if ts_start is not None:
                     ts_start = str(ts_start)
                     
                 evidence_list.append(EvidenceFragment(
                    source="visual",
                    content=f.get("content", ""),
                    timestamp_start=ts_start,
                    confidence=f.get("confidence", 0.0)
                ))
            
            # If there's a direct answer/caption, maybe add it as a high-confidence fragment or summary?
            if vision_result.get("answer"):
                 evidence_list.append(EvidenceFragment(
                    source="visual_summary",
                    content=f"Visual Answer: {vision_result.get('answer')}",
                    confidence=1.0
                ))
            
            if vision_result.get("caption"):
                 video.summary = vision_result.get("caption")

        # Also process transcript if we haven't already via vision (or do both? For now, let's do both if available)
        if full_transcript and not vision_result: 
             # Fallback to pure text extraction if vision failed or wasn't used
             try:
                prompt_text = load_prompt(
                    "watcher_extract.j2",
                    user_query=state["user_query"],
                    video_title=video.title,
                    transcript=full_transcript
                )
                response = llm.invoke([HumanMessage(content=prompt_text)])
                metrics = update_token_metrics(metrics, response)
                result = extract_json_from_text(response.content)
                for f in result.get("fragments", []):
                    ts_start = f.get("timestamp_start")
                    if ts_start is not None:
                        ts_start = str(ts_start)
                        
                    evidence_list.append(EvidenceFragment(
                        source="transcript",
                        content=f.get("content", ""),
                        timestamp_start=ts_start,
                        confidence=f.get("confidence", 0.0)
                    ))
             except Exception as e:
                 print(f"      ⚠️ Text extraction failed: {e}")

        # 5. Update State
        video.evidence = evidence_list
        video.status = "watched"

        if not evidence_list:
             video.relevance_reason = "No evidence found in transcript or visual analysis"

        print(f"      ✅ Extracted {len(evidence_list)} fragments.")
        logger.log("Watcher", "extracted", {
            "video_id": video.video_id,
            "fragment_count": len(evidence_list),
            "has_vision": bool(vision_result)
        })

        # 6. Record per-video step_log entry
        # Note: watcher.py uses a flat vision+transcript pipeline rather than
        # a 3-stage pyramidal structure. stage1_selected / stage2_frames_picked /
        # stage3_zoomed_ranges are adapted to what is observable here.
        videos_log.append(
            {
                "video_id": video.video_id,
                "title": video.title,
                "stage1_selected": True,  # video reached processing (passed candidate filter)
                "stage2_frames_picked": list(range(num_frames)) if vision_result else [],
                "stage3_zoomed_ranges": [],  # not applicable in this flat pipeline
                "relevant": bool(vision_result.get("relevant") if vision_result and isinstance(vision_result, dict) else evidence_list),
                "n_windows": len(evidence_list),
                "final_status": video.status,
            }
        )

        updated_videos[video.video_id] = video

    # Merge updates safely
    return {
        "video_store": {**state["video_store"], **updated_videos},
        "metrics": metrics,
        "step_log": [
            {
                "loop": state.get("loop_step", 0),
                "role": "watcher",
                "videos_processed": videos_log,
            }
        ],
    }
