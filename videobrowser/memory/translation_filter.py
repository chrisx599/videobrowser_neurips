"""Answer-mask filter for v2c distilled queries.

A candidate BM25 query is rejected if it contains the ground-truth answer
under word-boundary matching, after lowercasing both sides. Three forms
of the answer are checked: the raw token, the plural form (`+ "s"`), and
the possessive form (`+ "'s"`). The filter is the second leg of the
double-safeguard (the first being a hard constraint in the distill prompt).
"""
from __future__ import annotations

import re


def _build_pattern(answer: str) -> re.Pattern[str] | None:
    a = (answer or "").strip().lower()
    if not a:
        return None
    # Build alternation of {answer, answer+s, answer+'s}, each escaped, all
    # bracketed by word boundaries. We use re.escape to handle answers with
    # punctuation (rare but possible).
    forms = [a, a + "s", a + "'s"]
    alt = "|".join(re.escape(f) for f in forms)
    return re.compile(rf"\b(?:{alt})\b", re.IGNORECASE)


def answer_appears_in_query(query: str, answer: str) -> bool:
    """Return True if `answer` (or its plural / possessive form) appears
    in `query` under word-boundary matching."""
    pattern = _build_pattern(answer)
    if pattern is None:
        return False
    return pattern.search(query or "") is not None


def filter_queries_by_answer_mask(
    queries: list[str], *, answer: str,
) -> tuple[list[str], list[str]]:
    """Split `queries` into (kept, dropped) based on answer-mask matching."""
    pattern = _build_pattern(answer)
    if pattern is None:
        return list(queries), []
    kept: list[str] = []
    dropped: list[str] = []
    for q in queries:
        if pattern.search(q or ""):
            dropped.append(q)
        else:
            kept.append(q)
    return kept, dropped
