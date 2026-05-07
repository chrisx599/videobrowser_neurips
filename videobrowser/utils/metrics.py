"""Token-usage accounting helper used by every paradigm node.

Captures four dimensions per LLM response:

* ``input_tokens``         — total prompt tokens (matches OpenAI ``prompt_tokens``)
* ``output_tokens``        — completion tokens
* ``total_tokens``         — prompt + completion (sanity field; usually input+output)
* ``cached_input_tokens``  — subset of ``input_tokens`` that hit the prompt cache.
                             For OpenAI: ``prompt_tokens_details.cached_tokens``.
                             For Gemini's OpenAI-compat endpoint: same path.
                             For Anthropic: ``cache_read_input_tokens`` (when surfaced
                             via langchain's ``response_metadata``).  0 when the
                             provider doesn't report caching.

Cached tokens are billed at a discount; the cost-attribution scripts in
``scripts/`` use ``cached_input_tokens`` to apply provider-specific cached
input rates instead of the full input rate.
"""
from typing import Any, Dict


def _extract_cached_input_tokens(token_usage: Dict[str, Any]) -> int:
    """Best-effort cached-input-token extraction across providers.

    Recognised shapes (each provider populates one path):

    * OpenAI / Gemini-OpenAI-compat: ``token_usage['prompt_tokens_details']['cached_tokens']``
    * Some Gemini variants:          ``token_usage['cached_tokens']`` (top-level)
    * Anthropic via langchain:       ``token_usage['cache_read_input_tokens']``

    Unknown providers report 0 — safe fallback for cost computation
    (treats all input as uncached).
    """
    if not isinstance(token_usage, dict):
        return 0

    details = token_usage.get("prompt_tokens_details")
    if isinstance(details, dict):
        c = details.get("cached_tokens")
        if isinstance(c, (int, float)) and c > 0:
            return int(c)

    for key in ("cached_tokens", "cache_read_input_tokens"):
        c = token_usage.get(key)
        if isinstance(c, (int, float)) and c > 0:
            return int(c)

    return 0


def update_token_metrics(
    current_metrics: Dict[str, Any],
    response: Any,
    category: str = None,
) -> Dict[str, Any]:
    """Extract token usage from an LLM response and update the metrics dict.

    Args:
        current_metrics: Existing metrics dict from agent state.
        response: LLM response with ``.response_metadata['token_usage']``.
        category: Optional sub-bucket name (e.g. ``'watcher'``,
            ``'visual_describer'``).  When set, the same fields are
            accumulated into ``new_metrics[category]`` as a nested dict.

    Returns:
        New metrics dict with updated counters.  All four token fields
        (``input_tokens``, ``output_tokens``, ``total_tokens``,
        ``cached_input_tokens``) are always present at the top level.
    """
    if not current_metrics:
        current_metrics = {}

    token_usage: Dict[str, Any] = {}
    try:
        token_usage = response.response_metadata.get("token_usage", {}) or {}
    except Exception:
        token_usage = {}

    prompt_tokens = int(token_usage.get("prompt_tokens", 0) or 0)
    completion_tokens = int(token_usage.get("completion_tokens", 0) or 0)
    total_tokens = int(
        token_usage.get("total_tokens", prompt_tokens + completion_tokens) or 0
    )
    cached_input_tokens = _extract_cached_input_tokens(token_usage)

    new_metrics = current_metrics.copy()
    new_metrics["input_tokens"] = (
        int(new_metrics.get("input_tokens", 0)) + prompt_tokens
    )
    new_metrics["output_tokens"] = (
        int(new_metrics.get("output_tokens", 0)) + completion_tokens
    )
    new_metrics["total_tokens"] = (
        int(new_metrics.get("total_tokens", 0)) + total_tokens
    )
    new_metrics["cached_input_tokens"] = (
        int(new_metrics.get("cached_input_tokens", 0)) + cached_input_tokens
    )

    if category:
        cat_metrics = new_metrics.get(category) or {}
        if not isinstance(cat_metrics, dict):
            cat_metrics = {}
        cat_metrics["input_tokens"] = (
            int(cat_metrics.get("input_tokens", 0)) + prompt_tokens
        )
        cat_metrics["output_tokens"] = (
            int(cat_metrics.get("output_tokens", 0)) + completion_tokens
        )
        cat_metrics["total_tokens"] = (
            int(cat_metrics.get("total_tokens", 0)) + total_tokens
        )
        cat_metrics["cached_input_tokens"] = (
            int(cat_metrics.get("cached_input_tokens", 0)) + cached_input_tokens
        )
        new_metrics[category] = cat_metrics

    return new_metrics
