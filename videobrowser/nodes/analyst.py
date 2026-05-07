from videobrowser.core.state import AgentState
from videobrowser.utils.prompt_manager import load_prompt
from videobrowser.utils.metrics import update_token_metrics
from videobrowser.utils.logger import get_logger
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from videobrowser.utils.llm_factory import get_llm
from videobrowser.config import get_config
import time

load_dotenv()

llm = get_llm(node_name="analyst")
logger = get_logger()

def analyst_node(state: AgentState):
    print("📝 [Analyst] Synthesizing final report...")
    logger.log("Analyst", "start")
    
    # 1. Gather Videos
    # We primarily look for 'verified' videos, but if none, we might look at 'watched' 
    # (in case Checker forced a finish without verification).
    # For simplicity, let's take all videos that have some evidence.
    
    relevant_videos = []
    for v in state["video_store"].values():
        if v.evidence and len(v.evidence) > 0:
            relevant_videos.append({
                "id": v.video_id, # or a shorter index if preferred
                "title": v.title,
                "url": v.url,
                "evidence": v.evidence
            })
            
    # Check for text context
    text_context = state.get("text_search_context", [])

    if not relevant_videos and not text_context:
        msg = "I apologize, but I couldn't find any sufficient video evidence or text information to answer your specific query after several attempts."
        print(f"   -> {msg}")
        logger.log("Analyst", "fail", {"reason": "no_evidence"})
        return {
            "final_answer": msg,
            "step_log": [
                {
                    "loop": state.get("loop_step", 0),
                    "role": "analyst",
                    "final_answer": msg,
                    "has_evidence": False,
                    "evidence_summary": "No evidence found.",
                }
            ],
        }

    # 2. Construct Prompt
    # Retrieve formatting instructions from config
    config = get_config()
    format_instructions = config.prompts.analyst_format_instructions or ""

    prompt_text = load_prompt(
        "analyst_report.j2",
        user_query=state["user_query"],
        videos=relevant_videos,
        text_context=text_context,
        format_instructions=format_instructions
    )
    
    # 3. Invoke LLM
    metrics = state.get("metrics", {})
    try:
        response = llm.invoke([HumanMessage(content=prompt_text)])
        metrics = update_token_metrics(metrics, response)
        final_report = response.content
        print("   -> Report generated.")
        
        logger.log("Analyst", "generated_report", {
            "report_length": len(final_report)
        })
    except Exception as e:
        print(f"   ⚠️ Analyst failed: {e}")
        logger.log("Analyst", "error", {"error": str(e)}, level="ERROR")
        final_report = "Error generating report."
        
    # Calculate Final Metrics
    start_time = metrics.get("start_time")
    if start_time:
        end_time = time.time()
        duration = end_time - start_time
        print(f"\n📊 Performance Metrics:")
        print(f"   - Total Time: {duration:.2f}s")
        print(f"   - Input Tokens: {metrics.get('input_tokens', 0)}")
        print(f"   - Output Tokens: {metrics.get('output_tokens', 0)}")
        print(f"   - Total Tokens: {metrics.get('total_tokens', 0)}")
        
        logger.log("Analyst", "complete", {
            "duration": duration,
            "metrics": metrics
        })

    # 4. Build evidence summary for step_log (short, 2-3 lines)
    evidence_lines = []
    for v in relevant_videos[:3]:
        title = v.get("title", "Unknown")
        n_frags = len(v.get("evidence", []))
        evidence_lines.append(f"{title}: {n_frags} fragments")
    if len(relevant_videos) > 3:
        evidence_lines.append(f"... and {len(relevant_videos) - 3} more videos")
    evidence_summary = "; ".join(evidence_lines) if evidence_lines else "No video evidence used."
    if text_context and not evidence_lines:
        evidence_summary = f"{len(text_context)} text snippets used."

    # 5. Output
    return {
        "final_answer": final_report,
        "metrics": metrics,
        "step_log": [
            {
                "loop": state.get("loop_step", 0),
                "role": "analyst",
                "final_answer": final_report,
                "has_evidence": bool(relevant_videos or text_context),
                "evidence_summary": evidence_summary,
            }
        ],
    }

