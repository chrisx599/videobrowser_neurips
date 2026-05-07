import asyncio
import json
import os
import signal
import time
import concurrent.futures
from typing import Dict, Any
from tqdm import tqdm

# Import from the JIT workflow builder. JIT_BUILDER env var lets Table 3
# ablations swap in alternative builder modules (e.g. builder_ablation_blind)
# without duplicating the harness.
import importlib
_BUILDER_MODULE = os.environ.get("JIT_BUILDER", "builder")
build_graph = importlib.import_module(
    f"experiments.jit_paradigm.{_BUILDER_MODULE}"
).build_graph
from experiments.jit_paradigm.training_data import (
    append_jit_events,
    build_jit_experience_events,
)
from videobrowser.utils.llm_factory import get_llm
from videobrowser.utils.parser import extract_json_from_text
from videobrowser.utils.seed_recall import (
    compute_seed_recall,
    extract_distractor_video_ids,
    extract_seed_video_ids,
)
from langchain_core.messages import SystemMessage, HumanMessage
from videobrowser.config import get_config, load_config

def _kill_children():
    """Kill all child processes of the current process."""
    pid = os.getpid()
    try:
        # Read child PIDs from /proc
        import subprocess
        result = subprocess.run(
            ["pgrep", "-P", str(pid)], capture_output=True, text=True
        )
        for child_pid in result.stdout.strip().split("\n"):
            if child_pid:
                try:
                    os.kill(int(child_pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
    except Exception:
        # Fallback: kill process group but ignore error on self
        try:
            os.killpg(os.getpgid(pid), signal.SIGKILL)
        except Exception:
            pass


# --- Configuration ---
INPUT_FILE = os.environ.get("JIT_INPUT_FILE", "data/benchmark/videobrowsecomp/data_v4.5.jsonl")
_mode = os.environ.get("JIT_INPUT_MODE_FILTER", "offline")
INPUT_MODE_FILTER = _mode if _mode else None  # empty string = None (keep all)
MAX_WORKERS = int(os.environ.get("JIT_MAX_WORKERS", "16"))  # Proxy pool enables higher concurrency
CONFIG_PATH = os.environ.get("JIT_CONFIG_PATH", "experiments/jit_paradigm/config.yaml")
RUN_TAG = os.environ.get("JIT_RUN_TAG", "")

# --- Worker Function (Runs in a separate process) ---
def worker_task(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Independent worker function that runs in a separate process.
    """
    from videobrowser.tools.fetch_video import download_stats
    download_stats.reset()  # reset so we return per-episode delta, not cumulative
    # Load the specific config for this workflow
    load_config(CONFIG_PATH)

    row_id = row.get("row_id", "unknown")
    question = row["question"]
    ground_truth = row["answer"]
    
    # 1. Initialize local Graph and LLM
    try:
        app = build_graph()
        # Evaluator LLM
        llm = get_llm(node_name="analyst") 
    except Exception as e:
        return {
            "row_id": row_id,
            "error": f"Init error: {str(e)}",
            "is_correct": False
        }

    # 2. Define Execution Logic
    async def run_agent():
        inputs = {"user_query": question}

        thread_id = f"eval_proc_{row_id}"
        config = {"configurable": {"thread_id": thread_id}}

        final_state: Dict[str, Any] = {}
        # Accumulate every searcher's raw_candidates across loops for
        # retrieval-rank recall@k. The searcher overwrites raw_candidates each
        # loop, so we must capture it as it's emitted.
        retrieved_ids: list[str] = []
        seen_ids: set[str] = set()
        try:
            # stream_mode="values" yields the merged state after each node,
            # so we can both (a) capture raw_candidates from each loop and
            # (b) use the last yielded state as the final state for memory
            # event logging.
            async for state in app.astream(inputs, config=config, stream_mode="values"):
                final_state = state
                for cand in state.get("raw_candidates") or []:
                    if not isinstance(cand, dict):
                        continue
                    vid = cand.get("video_id") or cand.get("videoId") or cand.get("id")
                    if isinstance(vid, str) and vid and vid not in seen_ids:
                        seen_ids.add(vid)
                        retrieved_ids.append(vid)
        except Exception as e:
            final_state = {"final_answer": f"Error during execution: {str(e)}"}

        final_answer = final_state.get("final_answer", "No answer generated.")
        metrics = final_state.get("metrics", {})
        return final_answer, metrics, final_state, retrieved_ids

    async def run_judge(prediction: str):
        # Extract the actual answer from the agent's JSON response if possible
        clean_prediction = prediction
        try:
            pred_json = extract_json_from_text(prediction)
            if isinstance(pred_json, dict) and "Answer" in pred_json:
                clean_prediction = pred_json["Answer"]
        except Exception:
            # If parsing fails, use the full raw output
            pass
        # Short-circuit: empty answers cannot match any ground truth; smaller
        # judge LLMs have been observed to hallucinate is_correct=true on them.
        if not str(clean_prediction).strip():
            return {"is_correct": False}

        prompt = f"""
        Question: {question}

        Ground Truth: {ground_truth}

        Model Prediction: {clean_prediction}

        Evaluate if the Prediction matches the Ground Truth.
        Return JSON: {{"is_correct": true}} or {{"is_correct": false}}
        """
        try:
            response = await llm.ainvoke([
                SystemMessage(content="You are an expert evaluator."),
                HumanMessage(content=prompt)
            ])
            # Use robust parser
            return extract_json_from_text(response.content)
        except Exception:
            return {"is_correct": False}

    # 3. Execute
    start_time = time.time()
    
    metrics = {}
    final_state: Dict[str, Any] = {}
    retrieved_ids: list[str] = []
    try:
        # Run the async agent loop
        prediction, metrics, final_state, retrieved_ids = asyncio.run(run_agent())

        # Run the async judge
        eval_result = asyncio.run(run_judge(prediction))

        is_correct = eval_result.get("is_correct", False)

    except Exception as e:
        prediction = f"Critical Worker Error: {e}"
        is_correct = False

    # Emit experience events to memory.events_path (mirrors training pipeline).
    try:
        cfg = get_config()
        events_path = getattr(cfg.memory, "events_path", None)
        if cfg.memory.enabled and events_path and final_state:
            final_state["is_correct"] = is_correct
            events = build_jit_experience_events(row, final_state)
            if events:
                append_jit_events(events_path, events)
    except Exception as e:
        print(f"⚠️ event logging failed for row {row_id}: {e}")

    duration = time.time() - start_time

    # Extract specific metrics
    watcher_metrics = metrics.get("jit_watcher", {})
    analyst_metrics = metrics.get("jit_analyst", {})

    # Seed-video recall: retrieved = all search candidates across loops (in
    # retrieval-rank order, accumulated from each searcher emit); watched =
    # videos the JIT watcher actually selected + processed.
    video_store = final_state.get("video_store", {}) if isinstance(final_state, dict) else {}
    watched_ids = list(video_store.keys())
    # Guard: if streaming missed raw_candidates for any reason, fall back to
    # video_store keys so retrieved ⊇ watched still holds.
    if not retrieved_ids:
        retrieved_ids = list(watched_ids)
    seed_ids_list = extract_seed_video_ids(row)
    seed_block = compute_seed_recall(
        seed_ids=seed_ids_list,
        retrieved_ids=retrieved_ids,
        watched_ids=watched_ids,
        distractor_ids=extract_distractor_video_ids(row),
    )

    # Per-stage seed-recall ledger (set by builder_stage2_visual_selector via
    # step_log; absent in baselines). stage1 = K cands surviving metadata
    # selector, stage2 = M cands surviving visual selector. Accumulates the
    # union across all search loops.
    stage1_union: list[str] = []
    stage2_union: list[str] = []
    _seen_s1: set[str] = set()
    _seen_s2: set[str] = set()
    for entry in (final_state.get("step_log") or []):
        if not isinstance(entry, dict) or entry.get("role") != "watcher":
            continue
        for vid in entry.get("stage1_video_ids") or []:
            if vid and vid not in _seen_s1:
                _seen_s1.add(vid)
                stage1_union.append(vid)
        for vid in entry.get("stage2_video_ids") or []:
            if vid and vid not in _seen_s2:
                _seen_s2.add(vid)
                stage2_union.append(vid)

    seed_set = set(seed_ids_list)
    denom = len(seed_set) or 1
    s1_hits = seed_set & set(stage1_union)
    s2_hits = seed_set & set(stage2_union)
    stage_block = {
        "stage1_video_ids": stage1_union,
        "stage2_video_ids": stage2_union,
        "seed_in_stage1": bool(s1_hits) if seed_set else False,
        "seed_in_stage2": bool(s2_hits) if seed_set else False,
        "seed_recall_stage1": (len(s1_hits) / denom) if seed_set else 0.0,
        "seed_recall_stage2": (len(s2_hits) / denom) if seed_set else 0.0,
    }

    result = {
        "row_id": row_id,
        "question": question,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "is_correct": is_correct,
        "duration": duration,
        "input_tokens": metrics.get("input_tokens", 0),
        "output_tokens": metrics.get("output_tokens", 0),
        "total_tokens": metrics.get("total_tokens", 0),
        "watcher_tokens": watcher_metrics.get("total_tokens", 0),
        "analyst_tokens": analyst_metrics.get("total_tokens", 0),
        "whisper_audio_seconds": metrics.get("whisper_audio_seconds", 0.0),
        "_download_stats": {
            attr: getattr(download_stats, attr)
            for attr in vars(download_stats)
            if attr != "_lock"
        },
    }
    result.update(seed_block)
    result.update(stage_block)
    return result

# --- Main ---
def main():
    if not os.path.exists(INPUT_FILE):
        print(f"Error: Input file {INPUT_FILE} not found.")
        return

    # Generate timestamped output filename
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Ensure directory exists
    os.makedirs("data/benchmark/results", exist_ok=True)
    tag_suffix = f"_{RUN_TAG}" if RUN_TAG else ""
    output_file = f"data/benchmark/results/jit_evaluation_results_{timestamp}{tag_suffix}.jsonl"
    print(f"Config: {CONFIG_PATH}")
    print(f"Run tag: {RUN_TAG or '(none)'}")
    print(f"Results will be saved to: {output_file}")

    # Load Data
    data = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                row = json.loads(line)
                if INPUT_MODE_FILTER and row.get("mode") != INPUT_MODE_FILTER:
                    continue
                data.append(row)

    filter_note = f" (filtered by mode == {INPUT_MODE_FILTER!r})" if INPUT_MODE_FILTER else ""
    print(f"Loaded {len(data)} test cases{filter_note}. Starting pool with {MAX_WORKERS} workers.")

    # Load config FIRST so proxy validation uses the correct pool_path
    load_config(CONFIG_PATH)

    # Validate proxy pool before launching workers — dynamic IPs expire
    from videobrowser.tools.fetch_video import validate_proxy_pool, download_stats
    alive = validate_proxy_pool()
    if not alive:
        print("⚠️ No alive proxies — downloads will use direct connection.")

    results = []

    TASK_TIMEOUT = 600  # 10 min per task — prevents indefinite hangs on dead proxies

    # Use ProcessPoolExecutor for true parallelism
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS)
    try:
        future_to_row = {executor.submit(worker_task, row): row for row in data}

        for future in tqdm(concurrent.futures.as_completed(future_to_row), total=len(data)):
            row = future_to_row[future]
            try:
                result = future.result(timeout=TASK_TIMEOUT)

                # Merge per-worker download stats into main process
                worker_stats = result.pop("_download_stats", None)
                if worker_stats:
                    for attr, val in worker_stats.items():
                        cur = getattr(download_stats, attr, 0)
                        setattr(download_stats, attr, cur + val)

                results.append(result)

                with open(output_file, "a", encoding="utf-8") as f_out:
                    f_out.write(json.dumps(result, ensure_ascii=False) + "\n")

            except concurrent.futures.TimeoutError:
                print(f"⏰ Task {row.get('row_id', '?')} timed out after {TASK_TIMEOUT}s")
                results.append({
                    "row_id": row.get("row_id", "unknown"),
                    "question": row.get("question", ""),
                    "ground_truth": row.get("answer", ""),
                    "prediction": f"Timed out after {TASK_TIMEOUT}s",
                    "is_correct": False,
                    "duration": TASK_TIMEOUT,
                })
            except Exception as exc:
                print(f"Generated an exception: {exc}")

    except KeyboardInterrupt:
        print("\n⛔ Ctrl+C received — killing all workers...")
        executor.shutdown(wait=False, cancel_futures=True)
        # Kill child processes without killing ourselves
        _kill_children()
    finally:
        executor.shutdown(wait=False)

    # Final Stats
    correct_count = sum(1 for r in results if r.get("is_correct"))
    total = len(results)
    accuracy = (correct_count / total * 100) if total > 0 else 0
    
    total_duration = sum(r.get("duration", 0) for r in results)
    total_all_tokens = sum(r.get("total_tokens", 0) for r in results)
    total_watcher = sum(r.get("watcher_tokens", 0) for r in results)
    total_analyst = sum(r.get("analyst_tokens", 0) for r in results)

    print(f"\nEvaluation Complete.")
    print(f"Accuracy: {accuracy:.2f}% ({correct_count}/{total})")
    print(f"Total Inference Time (sum of all threads): {total_duration:.2f}s")
    print(f"Total Token Usage: {total_all_tokens}")
    print(f"Watcher Tokens: {total_watcher}")
    print(f"Analyst Tokens: {total_analyst}")
    print(f"Results saved to {output_file}")

    from videobrowser.tools.fetch_video import download_stats
    print(download_stats.summary())

if __name__ == "__main__":
    try:
        main()
    finally:
        # Restore terminal settings — subprocesses (yt-dlp, Puppeteer) can
        # disable echo if they crash mid-run.
        os.system("stty sane 2>/dev/null")
