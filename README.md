# Video-Browser — Supplementary Code

This bundle accompanies the paper *Video-Browser / Video-BrowseComp* (NeurIPS 2026 submission). It contains the source code and data needed to inspect (and, with appropriate model endpoints, reproduce) the main offline-track method reported in the paper.

The bundle is **scoped to our method only**. Baselines (ReAct, VideoAgent, VideoTree, DVD) are documented in the appendix but their code is omitted to keep the bundle small.

## Layout

```
supplementary_code/
├── README.md                                          this file
├── pyproject.toml                                     installable package metadata
├── requirements.txt                                   pinned Python dependencies
│
├── videobrowser/                                      core package (imported as `videobrowser.*`)
│   ├── config.py                                      YAML config loader + AppConfig dataclass
│   ├── core/state.py                                  AgentState / VideoResource definitions
│   ├── nodes/                                         LangGraph node implementations
│   │   ├── planner.py                                 § 4.3 Planner (with memory injection)
│   │   ├── searcher.py                                BM25 search dispatch
│   │   ├── watcher.py                                 reference Watcher (text Stage-1)
│   │   ├── selector.py                                Stage-1 candidate selector helpers
│   │   ├── checker.py                                 loop budget / termination
│   │   └── analyst.py                                 reference Analyst node
│   ├── memory/                                        Translation Memory (§ 4.2)
│   │   ├── bootstrap.py                               build_memory_runtime entry point
│   │   ├── translation_index.py                       BGE-M3 cosine index
│   │   ├── translation_retriever.py                   top-K retrieval over the bank
│   │   ├── translation_schemas.py                     MemoryCard schema
│   │   ├── injection.py                               Planner-side memory block rendering
│   │   ├── store.py / serializer.py                   JSONL bank persistence
│   │   └── (other files used by ablations / variants)
│   ├── prompts/                                       Jinja templates rendered at runtime
│   │   ├── planner.j2                                 Planner prompt
│   │   ├── memory_v2b_metadata_distill.j2             Translation Memory distillation prompt
│   │   ├── analyst_report.j2                          Analyst prompt
│   │   ├── watcher_extract.j2 / watcher_vision.j2     Watcher prompts
│   │   └── (additional ablation prompts)
│   ├── search_engine/                                 offline BM25 / BGE-M3 / hybrid retrievers
│   ├── tools/
│   │   ├── fetch_video.py                             yt-dlp / Whisper transcript fetch
│   │   ├── search_videos.py                           offline search facade
│   │   └── vision.py                                  frame extraction + grid mosaic
│   ├── hard_negatives/                                hard-negative pool construction (§ A.5)
│   ├── graph/builder.py                               legacy DAG builder helpers
│   └── utils/                                         logger / metrics / parser / llm_factory / cache
│
├── experiments/
│   ├── __init__.py
│   ├── jit_paradigm/
│   │   ├── builder_visual_stage1.py                   ★ MAIN method (Video-Browser / JIT)
│   │   ├── builder.py                                 supplies jit_checker_node, route_jit_checker
│   │   ├── builder_ablation_full_context.py           supplies jit_analyst_node (full-context Analyst)
│   │   ├── evaluate_multiprocess.py                   parallel eval driver
│   │   ├── training_data.py                           training-row helpers
│   │   └── configs/
│   │       ├── visual_stage1_hardneg.yaml             Qwen3-VL-8B config (Table 4 row "Ours, Qwen")
│   │       └── visual_stage1_hardneg_gemini.yaml      Gemini-3.1-Pro config (Table 4 row "Ours, Gemini")
│   └── experience_retrieval/
│       └── build_memory_v2b_metadata.py               offline Translation Memory build script
│
├── scripts/
│   ├── build_offline_pool.py                          rebuild data/offline_search/pool*.jsonl
│   └── build_hard_negatives.py                        rebuild data/offline_search/index_with_hard_negatives_notxt
│
└── data/                                              ~ 176 MB total, no video files
    ├── benchmark/
    │   └── videobrowsecomp/
    │       ├── data_v4.5.jsonl                        660 evaluation questions (full benchmark)
    │       └── train_candidates_1000.jsonl            1 000 training questions (used to distill Memory)
    ├── offline_search/
    │   ├── pool_with_hard_negatives.jsonl             12 455 video metadata rows (title/desc/channel/tags/transcript)
    │   ├── pool_inventory.jsonl                       per-video provenance (seed / cache / hard-negative)
    │   ├── pool_durations.jsonl                       ffprobe durations
    │   └── index_with_hard_negatives_notxt/           prebuilt BM25 index over title/desc/channel/tags
    │       ├── manifest.json
    │       └── bm25/{bm25.pkl, doc_ids.json, meta.json}
    └── training_runs/
        └── memory_v2b_metadata/
            ├── memory_bank.jsonl                      846 distilled MemoryCards (verified BM25 query + anchors)
            └── embeddings.npz                         BGE-M3 embeddings over the bank's index_text field
```

## Mapping to the paper

| Paper section                                  | Code entry point                                                                                       |
|------------------------------------------------|--------------------------------------------------------------------------------------------------------|
| § 4 Method overview                            | `experiments/jit_paradigm/builder_visual_stage1.py` (LangGraph wiring of Planner→Searcher→Watcher→Checker→Analyst) |
| § 4.2 Translation Memory                       | `videobrowser/memory/translation_*.py` (retrieval), `videobrowser/memory/injection.py` (planner injection) |
| § 4.2 Memory build (Appendix B.2)              | `experiments/experience_retrieval/build_memory_v2b_metadata.py`, prompt: `videobrowser/prompts/memory_v2b_metadata_distill.j2` |
| § 4.3 Sparse Watcher (Stage-1 visual selector) | `experiments/jit_paradigm/builder_visual_stage1.py::jit_visual_stage1_selector_node` (uses `videobrowser/tools/vision.py::extract_frames_as_grids`) |
| § 4.3 Iterative browsing                       | `experiments/jit_paradigm/builder.py::jit_checker_node` (loop budget 𝒯_max=3)                          |
| § 4.3 Analyst (full-context dense pass)        | `experiments/jit_paradigm/builder_ablation_full_context.py::jit_analyst_node`                          |
| § 5 Experiments / Table 4 (Qwen / Gemini rows) | configs `experiments/jit_paradigm/configs/visual_stage1_hardneg{,_gemini}.yaml`                        |
| § A.5 Hard-Negative Pool Construction          | `videobrowser/hard_negatives/`, `scripts/build_hard_negatives.py`                                       |
| § B.1 Implementation details                   | YAML configs in `experiments/jit_paradigm/configs/`                                                     |
| § B.2 Translation Memory construction          | `experiments/experience_retrieval/build_memory_v2b_metadata.py`                                        |

## Setting up

```bash
# from supplementary_code/
python -m venv .venv && source .venv/bin/activate
pip install -e .                       # installs the videobrowser package + deps
pip install -r requirements.txt        # pinned versions used in our runs
```

The package targets Python 3.10+. Key heavy dependencies are `langgraph`, `langchain-core`, `sentence-transformers` (for BGE-M3), `rank_bm25`, `pyav` (frame decoding), `Pillow`, `numpy`, `tqdm`.

## Required external services

The code expects the following services to be reachable. None are bundled (model weights and APIs are out of scope for code-supplementary uploads):

- **Multimodal LLM** — OpenAI-compatible endpoint at `llm.default.base_url` in the config. We ran Qwen3-VL-8B-Instruct via vLLM at `http://localhost:8025/v1` and Gemini-3.1-Pro via the public API.
- **Whisper transcription** — OpenAI-compatible endpoint at `transcript.base_url`. We ran `whisper-large-v3-turbo` at `http://localhost:8038/v1`. Most evaluation videos already have cached transcripts in `data/offline_search/pool_with_hard_negatives.jsonl`, so Whisper is only required if you re-fetch new videos.
- **BGE-M3 sentence-transformer** — local HuggingFace path, set via `memory.embedding.model_name` (default `/mnt/data/zhengyangliang/Models/BAAI/bge-m3`; reviewers should change this to their local path or `BAAI/bge-m3` to download from the Hub).
- **Cached video files** — to actually run the Watcher / Analyst the bundle must be paired with a local cache of the 12 455 candidate `.mp4` files. Cache root is `cache.base_dir` in the config (default `/mnt/data/zhengyangliang/videobrowser/data/cache`). The bundle does **not** include the video files themselves; the Searcher operates on the offline pool's metadata + transcript, and the agent will skip videos whose mp4 is missing.

## Running the main offline-track evaluation

```bash
# from supplementary_code/, after installing the package
export PYTHONPATH=.

# Qwen3-VL-8B (Table 4 row "Video-Browser, Qwen3-VL-8B"):
JIT_BUILDER=builder_visual_stage1 \
    python -m experiments.jit_paradigm.evaluate_multiprocess \
    --config experiments/jit_paradigm/configs/visual_stage1_hardneg.yaml \
    --tag visual_stage1_hardneg

# Gemini-3.1-Pro (Table 4 row "Video-Browser, Gemini-3.1-Pro"):
JIT_BUILDER=builder_visual_stage1 \
    python -m experiments.jit_paradigm.evaluate_multiprocess \
    --config experiments/jit_paradigm/configs/visual_stage1_hardneg_gemini.yaml \
    --tag visual_stage1_hardneg_gemini
```

Outputs land in `data/benchmark/results/jit_evaluation_results_<timestamp>_<tag>.jsonl`, one row per question with prediction, ground-truth, token usage, and per-stage retrieval signals.

## Rebuilding the Translation Memory bank

```bash
python -m experiments.experience_retrieval.build_memory_v2b_metadata \
    --train data/benchmark/videobrowsecomp/train_candidates_1000.jsonl \
    --pool data/offline_search/pool_with_hard_negatives.jsonl \
    --bm25-index data/offline_search/index_with_hard_negatives_notxt \
    --out data/training_runs/memory_v2b_metadata
```

This will overwrite `memory_bank.jsonl` and `embeddings.npz`. The bundled bank is what the paper's reported numbers were generated against.

## What is *not* in this bundle

- Cached video `.mp4` files (~1–2 TB on the original machine).
- Whisper transcripts beyond the per-video field embedded in the offline pool (audio is re-decoded if a Watcher / Analyst call needs more than the cached transcript).
- Ablation builders other than `builder_visual_stage1.py` (no-Stage-1, no-Stage-2, no-Memory variants), training-data collection scripts, and SFT / strategy-distillation pipelines.
- Baseline-agent code (ReAct, VideoAgent, VideoTree, DVD).
- Eval-result JSONL files for the main paper tables — those are listed by SHA in the paper.

## Path notes

YAML configs ship with absolute paths from the original training machine (`/mnt/data/zhengyangliang/...`). Reviewers running locally should edit `cache.base_dir`, `proxy.pool_path`, and `memory.embedding.model_name` to point at their own paths. Code-relative paths (`data/...`) work as-is from the bundle root.
