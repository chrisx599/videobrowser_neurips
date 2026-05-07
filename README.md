# Video-Browser — Supplementary Code

This bundle accompanies the paper *Video-Browser / Video-BrowseComp* (NeurIPS 2026 submission). It contains the source code and data needed to inspect (and, with appropriate model endpoints, reproduce) the main offline-track method reported in the paper.


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
└── data/                                              
    ├── benchmark/
        └── videobrowsecomp/
            ├── data_v4.5.jsonl                        660 evaluation questions (full benchmark)
            └── train_candidates_1000.jsonl            1 000 training questions (used to distill Memory)
    
```


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
- **BGE-M3 sentence-transformer** — local HuggingFace path, set via `memory.embedding.model_name`.
- **Cached video files** — to actually run the Watcher / Analyst the bundle must be paired with a local cache of the 12 455 candidate `.mp4` files. Cache root is `cache.base_dir` in the config. The bundle does **not** include the video files themselves; the Searcher operates on the offline pool's metadata + transcript, and the agent will skip videos whose mp4 is missing.

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


## Path notes

YAML configs ship with absolute paths from the original training machine. Reviewers running locally should edit `cache.base_dir`, `proxy.pool_path`, and `memory.embedding.model_name` to point at their own paths.
