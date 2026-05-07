from __future__ import annotations
from collections import defaultdict
from typing import Iterable, Mapping


_LAYER2_ORDER = {"NO": 0, "UNCERTAIN": 1, "YES": 2}
_RELEVANCE_ORDER = {"HIGH": 0, "MID": 1, "LOW": 2}


def _is_off_topic(row: Mapping) -> bool:
    """A candidate is off-topic if EITHER verifier judged its topical relevance LOW.

    Relevance fields may be absent (older verdict files) — treat absent as
    'unknown' (do not drop on this criterion).
    """
    for k in ("layer1_relevance", "layer2_relevance"):
        if row.get(k) == "LOW":
            return True
    return False


def _best_relevance_order(row: Mapping) -> int:
    """Lower-is-better composite relevance for sorting.

    Prefer rows where layer-2 (visual) judged HIGH; fall back to layer-1.
    Missing relevance is treated as MID (rank 1) so legacy rows aren't
    discriminated against in either direction.
    """
    l2 = _RELEVANCE_ORDER.get(row.get("layer2_relevance", "MID"), 1)
    l1 = _RELEVANCE_ORDER.get(row.get("layer1_relevance", "MID"), 1)
    return min(l2, l1)


def _sort_key(row: Mapping) -> tuple:
    return (
        -row["rewriter_overlap_score"],          # higher overlap first
        _best_relevance_order(row),              # HIGH < MID
        _LAYER2_ORDER.get(row["layer2_verdict"], 3),
        row["yt_rank_min"],                      # smaller rank first
    )


def select_top_n_per_question(
    rows: Iterable[Mapping],
    *,
    n: int,
) -> dict[str, list[dict]]:
    """Group rows by question_id and pick the top-N for each question.

    Drop rules (any one suffices):
      - layer2_verdict == "YES" (covert positive — would pollute eval)
      - layer1_relevance == "LOW" or layer2_relevance == "LOW" (off-topic
        viral / unrelated content; not a meaningful Stage-2 challenge)

    Sort key (descending priority):
      1. rewriter_overlap_score (desc) — multi-rewriter convergence
      2. best relevance across layers (HIGH < MID, missing = MID)
      3. layer2_verdict (NO < UNCERTAIN < YES)
      4. yt_rank_min (asc)
    """
    grouped: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r["layer2_verdict"] == "YES":
            continue
        if _is_off_topic(r):
            continue
        grouped[r["question_id"]].append(dict(r))
    out: dict[str, list[dict]] = {}
    for qid, items in grouped.items():
        items.sort(key=_sort_key)
        out[qid] = items[:n]
    return out
