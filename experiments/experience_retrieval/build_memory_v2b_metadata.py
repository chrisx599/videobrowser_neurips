"""Build planner-role memory bank v2b — metadata-distilled, BM25-verified search recipes.

Each card pairs:
  - index_text: the real train question (embedded for cosine match against user query at planner time)
  - payload_text: a verified BM25 query plus entity anchors mined from the seed video metadata

See docs/superpowers/specs/2026-04-28-jit-memory-redesign-design.md for the full design.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from jinja2 import Template

from videobrowser.memory.schemas import MemoryCard


# ---------------------------------------------------------------------------
# LLM distill response parsing
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.MULTILINE)

_ANCHOR_LIST_KEYS = ("entities", "salient_terms")
_ANCHOR_LIST_CAP = 4
_CHANNEL_SIGNAL_CAP = 60
_QUERY_WORD_CAP = 12
_REQUIRED_QUERIES = 5


@dataclass
class DistillOutput:
    queries: list[str]
    anchors: dict  # {"entities": list[str], "channel_signal": str, "salient_terms": list[str]}


def parse_distill_response(raw: str) -> DistillOutput:
    """Validate and normalize the LLM's distill JSON.

    Raises ValueError on malformed JSON, missing keys, or wrong query count.
    Truncates anchor lists to the documented caps so the payload stays bounded.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("distill response is empty")

    body = _FENCE_RE.sub("", raw).strip()
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"distill response is not valid JSON: {exc}") from exc

    if not isinstance(obj, dict):
        raise ValueError("distill response root must be a JSON object")

    queries = obj.get("queries")
    if not isinstance(queries, list) or len(queries) != _REQUIRED_QUERIES:
        raise ValueError(f"distill response must contain exactly 5 queries; got {queries!r}")
    queries = [str(q).strip() for q in queries if str(q).strip()]
    if len(queries) != _REQUIRED_QUERIES:
        raise ValueError("distill response must contain exactly 5 non-empty queries")

    for q in queries:
        if len(q.split()) > _QUERY_WORD_CAP:
            raise ValueError(f"query exceeds {_QUERY_WORD_CAP}-word limit: {q!r}")

    raw_anchors = obj.get("anchors") or {}
    if not isinstance(raw_anchors, dict):
        raise ValueError("distill anchors must be an object")

    anchors: dict = {}
    for key in _ANCHOR_LIST_KEYS:
        items = raw_anchors.get(key)
        if items is None:
            items = []
        elif not isinstance(items, list):
            raise ValueError(f"distill anchors.{key} must be a list")
        cleaned = [x.strip() for x in items if isinstance(x, str) and x.strip()]
        anchors[key] = cleaned[:_ANCHOR_LIST_CAP]

    channel_raw = raw_anchors.get("channel_signal")
    if channel_raw is None:
        channel = ""
    elif not isinstance(channel_raw, str):
        raise ValueError("distill anchors.channel_signal must be a string")
    else:
        channel = channel_raw
    anchors["channel_signal"] = channel.strip()[:_CHANNEL_SIGNAL_CAP]

    return DistillOutput(queries=queries, anchors=anchors)


# ---------------------------------------------------------------------------
# BM25 verification
# ---------------------------------------------------------------------------

@dataclass
class BestHit:
    query: str
    rank: int  # 1-based; ≤ topk by construction


def bm25_validate(engine, queries: list[str], target_video_id: str, topk: int) -> BestHit | None:
    """Run each candidate query through the offline BM25 engine; return the
    one whose target_video_id rank is smallest (≤ topk). Returns None if no
    candidate surfaces the target within topk.

    `engine` must expose `search(query: str, k: int) -> list[dict]` where each
    candidate dict has a top-level "id" key — this matches the production
    contract of `videobrowser.search_engine.engine.OfflineSearchEngine.search`,
    which calls `normalize_hit_to_candidate(...)` and returns plain dicts.
    """
    best: BestHit | None = None
    for query in queries:
        q = (query or "").strip()
        if not q:
            continue
        hits = engine.search(q, k=topk)
        for idx, hit in enumerate(hits, start=1):
            # BM25 returns each document at most once, so the first hit per
            # query is also the only hit; no need to keep scanning the rest.
            if isinstance(hit, dict) and hit.get("id") == target_video_id:
                if best is None or idx < best.rank:
                    best = BestHit(query=q, rank=idx)
                break
    return best


# ---------------------------------------------------------------------------
# Payload rendering
# ---------------------------------------------------------------------------

_QUESTION_PREVIEW_CAP = 80


def _truncate(text: str, cap: int) -> str:
    text = (text or "").strip().replace("\n", " ")
    if len(text) <= cap:
        return text
    return text[: cap - 1].rstrip() + "…"


def _join_anchors(anchors: dict) -> str:
    """Join entities + salient_terms with ' · ', deduplicating case-insensitively.
    channel_signal is omitted — it is a distillation hint, not a planner-visible
    anchor. Dedup preserves the first occurrence so entity ordering wins over
    a later salient_term that repeats it.
    """
    seen: set[str] = set()
    parts: list[str] = []
    for raw in (anchors.get("entities", []) or []) + (anchors.get("salient_terms", []) or []):
        if not isinstance(raw, str):
            continue
        s = raw.strip()
        if not s:
            continue
        key = s.lower()
        if key in seen:
            continue
        seen.add(key)
        parts.append(s)
    return " · ".join(parts)


def render_payload(
    *,
    question: str,
    category: str,
    best: BestHit | None,
    anchors: dict,
    verified: bool,
) -> str:
    """Render the planner-visible payload string for a v2b card.

    The card's `index_text` (the past question itself) is rendered separately
    by the injection layer, so the payload here carries only the verified
    BM25 query plus anchors — no redundant question preview, no rank noise.

    Verified case (best is not None and verified=True):
        Verified BM25 query: QUERY
        Key anchors: A1 · A2 · …

    Anchors-only case (best is None or verified=False):
        BM25 verification failed; suggested search terms: A1 · A2 · …

    `question` and `category` are kept in the signature for back-compat with
    callers and to support a future re-introduction of the head line if a
    downstream consumer ever needs it.
    """
    del question, category  # intentionally unused; kept for stable signature
    anchors_str = _join_anchors(anchors)

    if verified and best is not None:
        lines = [
            f"Verified BM25 query: {best.query}",
            f"Key anchors: {anchors_str}" if anchors_str else "Key anchors: (none)",
        ]
    else:
        anchors_part = anchors_str if anchors_str else "(none)"
        lines = [f"BM25 verification failed; suggested search terms: {anchors_part}"]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------

_PROMPT_PATH = Path(__file__).resolve().parents[2] / "videobrowser" / "prompts" / "memory_v2b_metadata_distill.j2"

_DESCRIPTION_CAP = 1500
_TRANSCRIPT_CAP = 800


def _flatten_transcript(transcript) -> str:
    """Pool records store transcript as either a plain string or a list of
    {start, end, text} segment dicts. Normalize to a single space-joined
    string before the cap is applied."""
    if transcript is None:
        return ""
    if isinstance(transcript, str):
        return transcript
    if isinstance(transcript, list):
        parts: list[str] = []
        for seg in transcript:
            if isinstance(seg, dict):
                t = seg.get("text")
                if isinstance(t, str) and t.strip():
                    parts.append(t.strip())
            elif isinstance(seg, str) and seg.strip():
                parts.append(seg.strip())
        return " ".join(parts)
    return ""


def _load_distill_template() -> "Template":
    """Module-level cache for the parsed Jinja template (read once)."""
    if _DISTILL_TEMPLATE_CACHE.get("template") is None:
        _DISTILL_TEMPLATE_CACHE["template"] = Template(_PROMPT_PATH.read_text(encoding="utf-8"))
    return _DISTILL_TEMPLATE_CACHE["template"]


_DISTILL_TEMPLATE_CACHE: dict = {}


def render_distill_prompt(
    *,
    question: str,
    title: str,
    description: str,
    channel: str,
    tags: list[str],
    transcript_snippet: str,
    feedback: str | None = None,
) -> str:
    """Render the v2b distill prompt. The Jinja Template is parsed once and cached at module level."""
    template = _load_distill_template()
    return template.render(
        question=(question or "").strip(),
        title=(title or "").strip(),
        description=(description or "")[:_DESCRIPTION_CAP].strip(),
        channel=(channel or "").strip(),
        tags=list(tags or []),
        transcript_snippet=_flatten_transcript(transcript_snippet)[:_TRANSCRIPT_CAP].strip(),
        feedback=(feedback or "").strip() or None,
    )


# ---------------------------------------------------------------------------
# Per-row orchestration
# ---------------------------------------------------------------------------


@dataclass
class BuildResult:
    card: MemoryCard
    status: str  # "ok" | "retry_ok" | "anchors_only" | "parse_failed"
    candidates_log: list[dict]  # [{"query": ..., "rank": int|None}, ...]
    n_attempts: int             # 1 or 2


_RETRY_FEEDBACK = (
    "prior queries did not surface the target video within the BM25 top-K; "
    "try more specific proper-noun entities and distinctive on-screen / spoken terms"
)


def _safe_distill(
    llm: Callable[[str], str],
    *,
    question: str,
    meta: dict,
    feedback: str | None,
) -> tuple[DistillOutput | None, str]:
    """Render the prompt, call llm, parse. Returns (output|None, raw_response)."""
    prompt = render_distill_prompt(
        question=question,
        title=meta.get("title", ""),
        description=meta.get("description", ""),
        channel=meta.get("channel", ""),
        tags=meta.get("tags", []) or [],
        transcript_snippet=meta.get("transcript", ""),
        feedback=feedback,
    )
    raw = llm(prompt)
    try:
        return parse_distill_response(raw), raw
    except ValueError:
        return None, raw


def _verify_with_log(
    engine, queries: list[str], target: str, topk: int
) -> tuple[list[dict], BestHit | None]:
    """Single BM25 pass: produce the audit log AND the rank-best hit in one go.

    For each candidate query, runs `engine.search(query, k=topk)` once and
    extracts both the per-query rank (for the audit log) and the rank-best
    hit (for the BestHit return). Avoids the double engine.search round-trips
    that calling bm25_validate twice (once for verify, once for logging)
    would incur for the same candidate list.
    """
    out: list[dict] = []
    best: BestHit | None = None
    for query in queries:
        q = (query or "").strip()
        if not q:
            continue
        rank: int | None = None
        for idx, hit in enumerate(engine.search(q, k=topk), start=1):
            if isinstance(hit, dict) and hit.get("id") == target:
                rank = idx
                break
        out.append({"query": q, "rank": rank})
        if rank is not None and (best is None or rank < best.rank):
            best = BestHit(query=q, rank=rank)
    return out, best


def build_v2b_card(
    *,
    row: dict,
    meta: dict,
    llm: Callable[[str], str],
    engine,
    topk: int = 3,
) -> BuildResult:
    """Build one v2b card for one train row.

    Pipeline: distill → BM25 verify → on miss, retry distill once with feedback
    → on still-miss, anchors-only card. On parse failure of both attempts,
    emit an anchors-only card with empty anchors and status='parse_failed'.
    """
    question = row["question"]
    category = row.get("category", "")
    difficulty = row.get("difficulty", "")
    target = row["video_id"]
    row_id = str(row.get("row_id", ""))

    # First pass
    out1, _ = _safe_distill(llm, question=question, meta=meta, feedback=None)
    cand_log: list[dict] = []
    best: BestHit | None = None
    anchors: dict = {"entities": [], "channel_signal": "", "salient_terms": []}

    if out1 is not None:
        log1, best = _verify_with_log(engine, out1.queries, target, topk)
        cand_log.extend(log1)
        anchors = out1.anchors

    if best is not None:
        status = "ok"
        verified = True
        n_attempts = 1
    else:
        # Retry once with feedback (or with same prompt if first pass failed to parse)
        out2, _ = _safe_distill(llm, question=question, meta=meta, feedback=_RETRY_FEEDBACK)
        n_attempts = 2
        if out2 is not None:
            log2, best = _verify_with_log(engine, out2.queries, target, topk)
            cand_log.extend(log2)
            if best is not None:
                status = "retry_ok"
                verified = True
                anchors = out2.anchors
            else:
                status = "anchors_only"
                verified = False
                # Prefer first-pass anchors if non-empty; otherwise second-pass.
                anchors = out1.anchors if out1 is not None else out2.anchors
        else:
            # Both passes failed to parse
            status = "parse_failed" if out1 is None else "anchors_only"
            verified = False
            anchors = out1.anchors if out1 is not None else {
                "entities": [], "channel_signal": "", "salient_terms": []
            }

    payload = render_payload(
        question=question,
        category=category,
        best=best,
        anchors=anchors,
        verified=verified,
    )

    card = MemoryCard(
        memory_id=f"v2b-meta-{row_id}",
        role="planner",
        phase_tag="exploration",
        gap_tags=[],
        outcome="success" if verified else "failure",
        memory_text=payload,
        index_text=question,
        payload_text=payload,
        tags={
            "category": category,
            "difficulty": difficulty,
            "memory_variant": "v2b_metadata",
            "verified": verified,
            "bm25_rank": best.rank if best else None,
        },
        source_trace_id=f"v2b-{row_id}",
        metadata={
            "source": "metadata_distill_bm25_verified",
            "row_id": row_id,
            "video_id": target,
            "question_preview": (question or "")[:140],
        },
    )
    return BuildResult(card=card, status=status, candidates_log=cand_log, n_attempts=n_attempts)


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------

import argparse
import sys


def _load_jsonl(path) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _build_pool_index(pool_path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in _load_jsonl(pool_path):
        vid = r.get("id")
        if vid:
            out[vid] = r
    return out


def run_build(
    *,
    train_path,
    pool_path,
    out_dir,
    llm: Callable[[str], str],
    engine,
    embedder,
    topk: int = 3,
    max_rows: int | None = None,
    progress_every: int = 0,
    concurrency: int = 1,
) -> dict:
    """End-to-end builder. Writes memory_bank.jsonl, extraction_log.jsonl,
    embeddings.npz under out_dir; returns a stats dict.

    Both bank and log are written line-by-line (streaming) with flush() after
    each row so a crash mid-build loses only the current row, not everything.

    Args:
        max_rows: Stop after writing this many cards (excludes skipped rows).
            Useful for dry-runs against large datasets.
        progress_every: When > 0, print one status line every N rows processed.
        concurrency: Number of parallel build_v2b_card workers (I/O bound on
            the LLM call). 1 = sequential (preserves prior behavior).
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from threading import Lock
    from experiments.experience_retrieval.build_memory_index import build_memory_index

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bank_path = out_dir / "memory_bank.jsonl"
    log_path = out_dir / "extraction_log.jsonl"
    emb_path = out_dir / "embeddings.npz"

    train_rows = _load_jsonl(train_path)
    pool_index = _build_pool_index(pool_path)

    status_counts: dict[str, int] = {}
    skipped = 0
    written = 0

    bank_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    bank_fh = bank_path.open("w", encoding="utf-8")
    log_fh = log_path.open("w", encoding="utf-8")
    write_lock = Lock()

    def _write_skip(row: dict, status: str) -> None:
        nonlocal skipped
        with write_lock:
            log_fh.write(json.dumps({
                "row_id": str(row.get("row_id", "")),
                "video_id": row.get("video_id"),
                "status": status,
                "n_attempts": 0,
                "best_rank": None,
                "candidates_log": [],
            }) + "\n")
            log_fh.flush()
            skipped += 1

    def _write_result(row: dict, result: BuildResult) -> None:
        nonlocal written
        with write_lock:
            status_counts[result.status] = status_counts.get(result.status, 0) + 1
            bank_fh.write(json.dumps(result.card.model_dump(mode="json"), ensure_ascii=True) + "\n")
            bank_fh.flush()
            log_fh.write(json.dumps({
                "row_id": str(row.get("row_id", "")),
                "video_id": row.get("video_id"),
                "status": result.status,
                "n_attempts": result.n_attempts,
                "best_rank": result.card.tags.get("bm25_rank"),
                "candidates_log": result.candidates_log,
            }) + "\n")
            log_fh.flush()
            written += 1

    # Pre-filter: separate definite skips (no LLM call) from buildable rows.
    buildable: list[tuple[int, dict, dict]] = []
    for i, row in enumerate(train_rows, start=1):
        if not (row.get("question") or "").strip():
            _write_skip(row, "skip_empty_question")
            continue
        meta = pool_index.get(row.get("video_id"))
        if meta is None:
            _write_skip(row, "skip_no_pool_match")
            continue
        buildable.append((i, row, meta))

    if max_rows is not None:
        buildable = buildable[:max_rows]

    def _worker(item):
        i, row, meta = item
        try:
            result = build_v2b_card(row=row, meta=meta, llm=llm, engine=engine, topk=topk)
        except Exception as e:
            print(f"  build error row_id={row.get('row_id', '?')}: {e}", flush=True)
            return i, row, None
        _write_result(row, result)
        return i, row, result

    try:
        if concurrency <= 1:
            for n, item in enumerate(buildable, start=1):
                i, row, result = _worker(item)
                if progress_every and (n % progress_every == 0) and result is not None:
                    print(
                        f"[{n}/{len(buildable)}] row_id={row.get('row_id', '?')} "
                        f"status={result.status}",
                        flush=True,
                    )
        else:
            done_count = 0
            with ThreadPoolExecutor(max_workers=concurrency) as pool:
                futures = [pool.submit(_worker, item) for item in buildable]
                for fut in as_completed(futures):
                    done_count += 1
                    try:
                        _, row, result = fut.result()
                    except Exception as e:
                        print(f"  worker exc: {e}", flush=True)
                        continue
                    if (
                        progress_every
                        and (done_count % progress_every == 0)
                        and result is not None
                    ):
                        print(
                            f"[{done_count}/{len(buildable)}] row_id={row.get('row_id', '?')} "
                            f"status={result.status}",
                            flush=True,
                        )
    finally:
        bank_fh.close()
        log_fh.close()

    build_memory_index(memory_bank_path=bank_path, output_path=emb_path, embedder=embedder)

    return {
        "total_rows": len(train_rows),
        "skipped_no_pool_match": skipped,
        "written": written,
        "status_counts": status_counts,
        "bank_path": str(bank_path),
        "log_path": str(log_path),
        "embedding_path": str(emb_path),
    }


def _build_default_llm_fn(
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
    api_key_env: str | None = None,
) -> Callable[[str], str]:
    """Wrap the OpenAI-compatible client into a single-prompt callable.

    For local vLLM servers, leave api_key_env=None (uses dummy "EMPTY" key).
    For hosted endpoints (e.g. Gemini's OpenAI-compat path), pass the env
    var name that holds the API key (e.g. "GEMINI_API_KEY").
    """
    import os
    from openai import OpenAI

    api_key = "EMPTY"
    if api_key_env:
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise SystemExit(f"Env var {api_key_env} is not set")

    client = OpenAI(base_url=base_url, api_key=api_key)

    def _call(prompt: str) -> str:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()

    return _call


def main() -> None:
    p = argparse.ArgumentParser(description="Build v2b metadata-distilled BM25-verified memory bank.")
    p.add_argument("--train", default="data/benchmark/videobrowsecomp/train_candidates_1000.jsonl")
    p.add_argument("--pool", default="data/offline_search/pool.jsonl")
    p.add_argument("--out-dir", default="data/training_runs/memory_v2b_metadata")
    p.add_argument("--bm25-topk", type=int, default=3)
    p.add_argument("--llm-base-url", default="http://localhost:8025/v1")
    p.add_argument("--llm-model", default="Qwen3-VL-8B-Instruct")
    p.add_argument("--llm-temperature", type=float, default=0.4)
    p.add_argument("--llm-max-tokens", type=int, default=512)
    p.add_argument("--llm-api-key-env", default=None,
                   help="Env var name for API key (e.g. GEMINI_API_KEY). "
                        "Leave unset for local vLLM servers.")
    p.add_argument("--embedding-model",
                   default="models/BAAI/bge-m3")
    p.add_argument("--embedding-url", default=None,
                   help="If set, embed via the BGE-M3 vLLM HTTP server "
                        "(e.g. http://localhost:8030/v1/embeddings) instead of "
                        "loading SentenceTransformer locally. Overrides --embedding-model.")
    p.add_argument("--embedding-served-name", default="BAAI/bge-m3",
                   help="served-model-name on the vLLM server (used only with --embedding-url).")
    p.add_argument("--embedding-batch", type=int, default=64,
                   help="HTTP embed batch size (used only with --embedding-url).")
    p.add_argument("--max-rows", type=int, default=None,
                   help="Stop after writing this many cards (excludes pool-miss rows).")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Parallel build_v2b_card workers (I/O bound on LLM API).")
    p.add_argument("--progress-every", type=int, default=10,
                   help="Print progress every N rows. Set 0 to disable.")
    p.add_argument("--config-path",
                   default="experiments/jit_paradigm/configs/memory_v2b_metadata.yaml",
                   help="YAML loaded for the BM25 engine (must enable search.offline).")
    args = p.parse_args()

    from videobrowser.config import load_config
    from videobrowser.search_engine.engine import get_default_engine

    # The BM25 engine reads search.offline from videobrowser/config.py's
    # AppConfig. We load the JIT eval config because the root config.yaml has
    # search.offline.enabled=False by default. The --pool CLI arg feeds only
    # the metadata-lookup index used for the LLM prompt; it does NOT override
    # which pool BM25 searches (that lives in the loaded YAML).
    cfg = load_config(args.config_path)
    engine = get_default_engine(cfg)
    if engine is None:
        print("❌ Offline BM25 engine unavailable — refusing to build unverified bank.", file=sys.stderr)
        sys.exit(2)

    llm = _build_default_llm_fn(
        base_url=args.llm_base_url,
        model=args.llm_model,
        temperature=args.llm_temperature,
        max_tokens=args.llm_max_tokens,
        api_key_env=args.llm_api_key_env,
    )
    if args.embedding_url:
        from experiments.local_inference.http_embedder import HttpEmbeddingBackend
        embedder = HttpEmbeddingBackend(
            url=args.embedding_url,
            model=args.embedding_served_name,
            batch_size=args.embedding_batch,
        )
        print(f"[embedder] HTTP backend → {args.embedding_url} (model={args.embedding_served_name})")
    else:
        from sentence_transformers import SentenceTransformer
        embedder = SentenceTransformer(args.embedding_model)

    stats = run_build(
        train_path=args.train,
        pool_path=args.pool,
        out_dir=args.out_dir,
        llm=llm,
        engine=engine,
        embedder=embedder,
        topk=args.bm25_topk,
        max_rows=args.max_rows,
        progress_every=args.progress_every,
        concurrency=args.concurrency,
    )
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
