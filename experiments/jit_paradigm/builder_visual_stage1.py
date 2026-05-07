"""JIT builder: VISUAL Stage 1 selector (no Stage 2).

Watcher pipeline per loop:
  1. Pre-filter already-seen candidates from BM25's raw_candidates.
  2. For each remaining candidate, build N grids (default 2 × 9 frames @ 256px)
     by uniform-sampling the cached video — disk-cached so a candidate is
     decoded once across the entire experiment suite.
  3. One multimodal LLM call: title/description/channel/tags + grid images for
     all N candidates → pick top_k indices.
  4. For each picked video: download, fetch timestamped transcript, store as
     ``VideoResource`` in ``video_store`` (no Stage 2 windowing — Analyst will
     extract dense frames over the full duration on its own).

Then standard checker → analyst (`jit_analyst_node` from
``builder_ablation_full_context``) — identical to ``builder_ablation_no_stage2``
downstream so the diff is purely Stage 1 text-vs-visual.

Per-step extras logged for the eval harness (``stage1_video_ids``,
``had_visuals``) so seed-recall numbers stay comparable to the baseline.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

# Pin decord to 1 thread per call before importing modules that load decord —
# without this the per-VideoReader thread pool * concurrent calls thrashes.
os.environ.setdefault("DECORD_NUM_THREADS", "1")

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from videobrowser.core.state import AgentState, VideoResource
from videobrowser.utils.parser import extract_youtube_id, extract_json_from_text
from videobrowser.utils.llm_factory import get_llm
from videobrowser.tools.fetch_video import fetch_transcript_with_timestamps, download_video_file
from videobrowser.tools.vision import extract_frames_as_grids
from videobrowser.config import load_config, get_config
from videobrowser.memory.bootstrap import build_memory_runtime
from videobrowser.utils.metrics import update_token_metrics
from videobrowser.utils.logger import get_logger

from videobrowser.nodes.planner import planner_node
from videobrowser.nodes.searcher import searcher_node

from experiments.jit_paradigm.builder_ablation_full_context import jit_analyst_node
from experiments.jit_paradigm.builder import jit_checker_node, route_jit_checker

try:
    from decord import VideoReader, cpu
except ImportError:
    VideoReader = None


# Stage 1 visual defaults — kept inside the builder so configs only override
# what's relevant. 2 grids × 9 frames @ 256px ≈ 18 frames + ~450 KB / video.
DEFAULT_S1_GRIDS = 2
DEFAULT_S1_FRAMES_PER_GRID = 9
DEFAULT_S1_CELL_SIZE = 256

VIDEO_CACHE_DIR = Path("data/cache/videos")
GRID_CACHE_DIR = Path("data/cache/grids")
GRID_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _grid_cache_path(video_id: str, n_grids: int, frames_per_grid: int, cell: int) -> Path:
    return GRID_CACHE_DIR / f"{video_id}__g{n_grids}_f{frames_per_grid}_c{cell}.json"


def _extract_grids_cached(video_id: str, n_grids: int, frames_per_grid: int, cell: int):
    """Return list of grid dicts; reuse on-disk cache.

    Cache file layout matches scripts/eval_visual_selector_top1.py so both
    pipelines share entries.
    """
    cache_path = _grid_cache_path(video_id, n_grids, frames_per_grid, cell)
    if cache_path.exists():
        try:
            with cache_path.open("r") as f:
                grids = json.load(f)
            if isinstance(grids, list) and grids:
                return grids
        except Exception:
            pass

    p = VIDEO_CACHE_DIR / f"{video_id}.mp4"
    if not p.exists():
        return []
    try:
        grids = extract_frames_as_grids(
            str(p),
            num_grids=n_grids,
            frames_per_grid=frames_per_grid,
            cell_size=(cell, cell),
        ) or []
    except Exception as exc:
        print(f"      ⚠️ S1-visual grid extract failed for {video_id}: {exc}")
        return []

    if grids:
        try:
            tmp = cache_path.with_suffix(".json.tmp")
            with tmp.open("w") as f:
                json.dump(grids, f)
            tmp.replace(cache_path)
        except Exception as exc:
            print(f"      ⚠️ S1-visual grid cache write failed for {video_id}: {exc}")
    return grids


def _short(s, n=240):
    if not s:
        return ""
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def _build_s1_visual_content(user_query: str, candidates: list[dict],
                              cand_grids: list[list[dict]], top_k: int,
                              n_grids: int, frames_per_grid: int) -> list[dict]:
    rows = int(frames_per_grid ** 0.5)
    head = (
        f'You are screening video search results.\n'
        f'User Query: "{user_query}"\n\n'
        f'You see {len(candidates)} BM25-retrieved candidates. For each, the '
        f'metadata is shown below; up to {n_grids} {rows}x{rows} grids '
        f'({frames_per_grid} frames each, sampled uniformly across the video '
        f'in temporal order) are attached. Each cell is labelled with its '
        f'timestamp.\n\n'
        f'Pick the {top_k} most relevant candidate(s) whose metadata + visual '
        f'content most plausibly contain the answer.\n\n'
        f'Respond with ONLY a JSON list of {top_k} integer candidate indices, '
        f'best first. Example: [3, 0, 7]'
    )
    content: list[dict] = [{"type": "text", "text": head}]
    for i, cand in enumerate(candidates):
        title = _short(cand.get("title"), 200)
        desc = _short(cand.get("snippet") or cand.get("description"), 240)
        ch = _short(cand.get("channel"), 60)
        meta = f"\n[Cand {i}] Title: {title}"
        if ch:
            meta += f" | Channel: {ch}"
        if desc:
            meta += f"\n  Desc: {desc}"
        content.append({"type": "text", "text": meta})
        grids = cand_grids[i]
        if not grids:
            content.append({"type": "text", "text": "  (no visual data)"})
            continue
        for gi, g in enumerate(grids):
            ts = g["timestamps"]
            content.append({
                "type": "text",
                "text": f"  [Cand {i} · Grid {gi+1}/{len(grids)} · "
                        f"t={ts[0]:.0f}–{ts[-1]:.0f}s]",
            })
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{g['image']}"},
            })
    return content


def jit_watcher_node_visual_stage1(state: AgentState):
    """Stage 1 = VISUAL selector (metadata + grids per candidate). No Stage 2."""
    logger = get_logger()
    logger.log("JITWatcher", "start", {"variant": "visual_stage1"})
    print("👁️  [JIT Watcher — Visual Stage 1] grid-aware metadata selector...")
    config = get_config()
    metrics = state.get("metrics", {})
    video_store = state.get("video_store", {})

    raw_candidates = state.get("raw_candidates", [])
    if not raw_candidates:
        return {
            "video_store": video_store,
            "step_log": [
                {"loop": state.get("loop_step", 0), "role": "watcher", "videos_processed": []}
            ],
        }

    user_query = state.get("user_query", "")
    llm = get_llm(node_name="watcher")
    top_k = config.selector.top_k

    n_grids = int(getattr(config.watcher, "num_grids", None) or DEFAULT_S1_GRIDS)
    frames_per_grid = int(getattr(config.watcher, "frames_per_grid", None) or DEFAULT_S1_FRAMES_PER_GRID)
    cell = int(getattr(config.watcher, "grid_cell_size", None) or DEFAULT_S1_CELL_SIZE)

    # Pre-filter already-seen candidates
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
        cand = raw.copy()
        cand["url"] = url
        cand["video_id"] = vid or ""
        valid_candidates.append(cand)

    if skipped_seen:
        print(f"   -> Pre-filtered {skipped_seen} already-seen candidates.")
    if not valid_candidates:
        return {
            "video_store": video_store,
            "step_log": [
                {"loop": state.get("loop_step", 0), "role": "watcher", "videos_processed": []}
            ],
        }

    # Build grids for every candidate (cache makes repeats free)
    cand_ids = [c["video_id"] for c in valid_candidates]
    print(f"   -> Stage 1 visual: extracting {n_grids} grids × {frames_per_grid} frames "
          f"@ {cell}px for {len(valid_candidates)} candidates "
          f"(cache: {GRID_CACHE_DIR})")
    cand_grids: list[list[dict]] = []
    n_with_visuals = 0
    for c in valid_candidates:
        gs = _extract_grids_cached(c["video_id"], n_grids, frames_per_grid, cell) if c["video_id"] else []
        cand_grids.append(gs)
        if gs:
            n_with_visuals += 1
    print(f"   -> grids ready for {n_with_visuals}/{len(valid_candidates)} candidates")

    # One multimodal LLM call to pick top-K
    target_videos: list[dict] = []
    if n_with_visuals == 0:
        print("      ⚠️ No grids built; falling back to first top_k candidates.")
        target_videos = valid_candidates[: top_k]
        s1_pick_indices: list[int] = list(range(min(top_k, len(valid_candidates))))
    else:
        content = _build_s1_visual_content(
            user_query, valid_candidates, cand_grids, top_k, n_grids, frames_per_grid,
        )
        try:
            response = llm.invoke([HumanMessage(content=content)])
            metrics = update_token_metrics(metrics, response, category="jit_selector")
            picked = extract_json_from_text(response.content)
            s1_pick_indices = []
            if isinstance(picked, list):
                for x in picked:
                    if isinstance(x, int) and 0 <= x < len(valid_candidates) and x not in s1_pick_indices:
                        s1_pick_indices.append(x)
                    if len(s1_pick_indices) >= top_k:
                        break
            elif isinstance(picked, dict):
                for x in picked.values():
                    if isinstance(x, int) and 0 <= x < len(valid_candidates) and x not in s1_pick_indices:
                        s1_pick_indices.append(x)
                    if len(s1_pick_indices) >= top_k:
                        break
            if not s1_pick_indices:
                s1_pick_indices = list(range(min(top_k, len(valid_candidates))))
            target_videos = [valid_candidates[i] for i in s1_pick_indices]
        except Exception as exc:
            print(f"      ⚠️ Stage 1 visual call failed: {exc}; falling back to top {top_k}.")
            s1_pick_indices = list(range(min(top_k, len(valid_candidates))))
            target_videos = [valid_candidates[i] for i in s1_pick_indices]

    print(f"   -> Stage 1 visual picked {len(target_videos)} candidates: "
          f"{[v.get('video_id') for v in target_videos]}")

    # For each picked video: download (for analyst's frame extraction +
    # whisper duration accounting) and fetch transcript. Store as VideoResource.
    videos_log_for_step: list[dict] = []
    for i, video in enumerate(target_videos):
        video_url = video["url"]
        video_id = video.get("video_id") or extract_youtube_id(video_url) or f"vid_{i}"
        if video_id in video_store:
            continue

        video_path = None
        try:
            video_path = download_video_file(video_url)
        except Exception as exc:
            print(f"      ⚠️ Download error for {video_id}: {exc}")

        transcript_text = ""
        try:
            transcript_segments = fetch_transcript_with_timestamps(video_url)
            if transcript_segments:
                if (
                    config.transcript.provider == "whisper"
                    and VideoReader
                    and video_path
                    and os.path.exists(video_path)
                ):
                    try:
                        vr = VideoReader(video_path, ctx=cpu(0))
                        duration = len(vr) / vr.get_avg_fps()
                        metrics["whisper_audio_seconds"] = (
                            metrics.get("whisper_audio_seconds", 0.0) + duration
                        )
                    except Exception as exc:
                        print(f"      ⚠️ duration probe failed: {exc}")
                lines = []
                for seg in transcript_segments:
                    lines.append(
                        f"[{seg.get('start', 0):.1f}s - {seg.get('end', 0):.1f}s] "
                        f"{seg.get('text', '')}"
                    )
                transcript_text = "\n".join(lines)
        except Exception as exc:
            print(f"      ⚠️ Transcript error for {video_id}: {exc}")

        resource = VideoResource(
            video_id=video_id,
            title=video.get("title", "Unknown"),
            url=video_url,
            duration=video.get("duration", "Unknown"),
            status="verified",
            summary="",  # no Stage 2 windowing
            transcript=transcript_text,
        )
        video_store[video_id] = resource

        videos_log_for_step.append({
            "video_id": video_id,
            "title": video.get("title", "Unknown"),
            "stage1_selected": True,
            "stage2_frames_picked": [],
            "stage3_zoomed_ranges": [],
            "relevant": True,
            "n_windows": 0,
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
                # Extras for the eval harness's per-stage seed-recall ledger
                "stage1_video_ids": [c["video_id"] for c in valid_candidates],
                "stage1_picked_video_ids": [v.get("video_id", "") for v in target_videos],
                "n_with_visuals": n_with_visuals,
            }
        ],
    }


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("planner", planner_node)
    workflow.add_node("searcher", searcher_node)
    workflow.add_node("watcher", jit_watcher_node_visual_stage1)
    workflow.add_node("checker", jit_checker_node)
    workflow.add_node("analyst", jit_analyst_node)

    workflow.add_edge(START, "planner")
    workflow.add_edge("planner", "searcher")
    workflow.add_edge("searcher", "watcher")
    workflow.add_edge("watcher", "checker")
    workflow.add_conditional_edges(
        "checker",
        route_jit_checker,
        {"planner": "planner", "analyst": "analyst"},
    )
    workflow.add_edge("analyst", END)

    return workflow.compile(checkpointer=MemorySaver())


if __name__ == "__main__":
    config_obj = load_config("experiments/jit_paradigm/config.yaml")
    memory_runtime = build_memory_runtime(config_obj)
    app = build_graph()
    print("[visual_stage1] graph built. Run via the standard JIT eval driver.")
