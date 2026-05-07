"""Seed-video recall helpers for per-paradigm evaluation harnesses.

Each benchmark row's `videos` field is a list of dicts carrying `role`,
`video_id`, `url`, etc. Rows with `role == "seed"` are the ground-truth
videos that contain the answer; the rest are distractors. Eval harnesses
use the helpers below to:

1. Extract seed / distractor ids from the input row.
2. Compute whether the paradigm's retrieved / watched video sets covered
   those seeds, and fold the result into the output JSONL row.
"""
from __future__ import annotations

from typing import Any, Iterable


# Video role semantics in the offline split:
#   - `source` (level1 / level2, 389 rows): the single answer-bearing video.
#   - `seed`   (level3 only, 61 rows): the primary answer-bearing video.
#   - `positive` (level3 only, 123 entries): supporting-evidence videos.
#     level3 questions are multi-hop — the answer requires synthesis across
#     `seed` + one or more `positive` videos. To avoid under-reporting
#     retrieval success on level3, we treat all three as GT.
#   - `negative` (level3 only, 277 entries): hard-negative distractors, not GT.
SEED_ROLES: frozenset[str] = frozenset({"seed", "source", "positive"})
DISTRACTOR_ROLES: frozenset[str] = frozenset({"negative"})


def _dedup_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if not item:
            continue
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def _collect_video_ids(row: dict[str, Any], roles: Iterable[str]) -> list[str]:
    role_set = set(roles)
    videos = row.get("videos") or []
    if not isinstance(videos, list):
        return []
    ids: list[str] = []
    for entry in videos:
        if not isinstance(entry, dict):
            continue
        if entry.get("role") not in role_set:
            continue
        vid = entry.get("video_id")
        if isinstance(vid, str) and vid:
            ids.append(vid)
    return _dedup_preserve_order(ids)


def extract_seed_video_ids(row: dict[str, Any]) -> list[str]:
    """Return the GT seed video ids for a benchmark row.

    Filters `row["videos"]` to entries whose role is in `SEED_ROLES`
    (``seed`` or ``source``) and returns their `video_id` values,
    dedup'd preserving first-seen order.
    """
    return _collect_video_ids(row, SEED_ROLES)


def extract_distractor_video_ids(row: dict[str, Any]) -> list[str]:
    """Return GT distractor video ids (role in DISTRACTOR_ROLES)."""
    return _collect_video_ids(row, DISTRACTOR_ROLES)


DEFAULT_RECALL_KS: tuple[int, ...] = (1, 3, 5, 10, 20)


def recall_at_k(seed_ids: Iterable[str], ranked_ids: Iterable[str], k: int) -> float:
    """Fraction of seed ids that appear within the first `k` of `ranked_ids`.

    Assumes `ranked_ids` is already ordered by retrieval rank (first-surfaced
    first). `k <= 0` → 0.0; empty seeds → 0.0.
    """
    seeds = _dedup_preserve_order(seed_ids)
    if not seeds or k <= 0:
        return 0.0
    ranked = _dedup_preserve_order(ranked_ids)
    top_k = set(ranked[:k])
    hits = sum(1 for s in seeds if s in top_k)
    return hits / len(seeds)


def compute_recall_at_k(
    seed_ids: Iterable[str],
    ranked_ids: Iterable[str],
    k_values: Iterable[int] = DEFAULT_RECALL_KS,
) -> dict[str, float]:
    """Return {str(k): recall@k} for each k. Keys are stringified for JSON."""
    seeds_cached = _dedup_preserve_order(seed_ids)
    ranked_cached = _dedup_preserve_order(ranked_ids)
    return {str(k): recall_at_k(seeds_cached, ranked_cached, k) for k in k_values}


def compute_seed_recall(
    seed_ids: Iterable[str],
    retrieved_ids: Iterable[str],
    watched_ids: Iterable[str],
    distractor_ids: Iterable[str] | None = None,
    recall_ks: Iterable[int] = DEFAULT_RECALL_KS,
) -> dict[str, Any]:
    """Compute the per-row seed recall block.

    Returns the canonical eight-field dict plus id lists and a nested
    ``seed_recall_at_k`` map so harnesses can fold the whole thing into the
    output row with ``row.update(...)``.

    All id iterables are dedup'd preserving first-seen order. Overall recall
    is ``len(seed ∩ set) / len(seed)`` or ``0.0`` when ``seed`` is empty.
    ``recall_at_k`` assumes ``retrieved_ids`` is already ordered by
    retrieval rank (which is how every paradigm harness populates it).
    """
    seeds = _dedup_preserve_order(seed_ids)
    retrieved = _dedup_preserve_order(retrieved_ids)
    watched = _dedup_preserve_order(watched_ids)
    distractors = _dedup_preserve_order(distractor_ids or [])

    seed_set = set(seeds)
    retrieved_hits = seed_set & set(retrieved)
    watched_hits = seed_set & set(watched)

    denom = len(seeds) or 1
    return {
        "seed_video_ids": seeds,
        "distractor_video_ids": distractors,
        "retrieved_video_ids": retrieved,
        "watched_video_ids": watched,
        "seed_in_retrieved": bool(retrieved_hits) if seeds else False,
        "seed_in_watched": bool(watched_hits) if seeds else False,
        "seed_recall_retrieved": (len(retrieved_hits) / denom) if seeds else 0.0,
        "seed_recall_watched": (len(watched_hits) / denom) if seeds else 0.0,
        "seed_recall_at_k": compute_recall_at_k(seeds, retrieved, recall_ks),
    }
