from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Sequence

from langchain_core.messages import HumanMessage, SystemMessage

from videobrowser.memory.schemas import MemoryCard
from videobrowser.utils.llm_factory import get_llm
from videobrowser.utils.parser import extract_json_from_text
from videobrowser.utils.prompt_manager import load_prompt


RecommendedAction = Literal["apply", "skip", "loop"]


@dataclass
class CriticResult:
    ranked_memory_ids: list[str]
    applicability_scores: dict[str, float]
    critic_confidence: float
    recommended_action: RecommendedAction
    rationale: str
    parse_failed: bool = False
    tokens: dict[str, int] | None = None


def _episode_cache_key(
    role: str,
    query_text: str,
    candidate_ids: Iterable[str],
) -> str:
    q_hash = hashlib.sha1(query_text.encode("utf-8")).hexdigest()[:16]
    id_tuple = tuple(sorted(candidate_ids))
    id_hash = hashlib.sha1(",".join(id_tuple).encode("utf-8")).hexdigest()[:16]
    return f"{role}:{q_hash}:{id_hash}"


def _fallback_result(
    candidates: Sequence[MemoryCard],
    fallback_confidence: float,
) -> CriticResult:
    ids = [c.memory_id for c in candidates]
    return CriticResult(
        ranked_memory_ids=ids,
        applicability_scores={mid: 1.0 for mid in ids},
        critic_confidence=float(fallback_confidence),
        recommended_action="apply",
        rationale="fallback: critic parse/call failed; retaining cosine order.",
        parse_failed=True,
    )


def _coerce_action(value: Any) -> RecommendedAction:
    if isinstance(value, str) and value.lower() in {"apply", "skip", "loop"}:
        return value.lower()  # type: ignore[return-value]
    return "apply"


def _coerce_float(value: Any, default: float = 0.5) -> float:
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if x != x:  # NaN
        return default
    return max(0.0, min(1.0, x))


class MemoryCritic:
    """Frozen-LLM reranker + applicability critic.

    Ranks the top-N BGE-M3 proposals and emits applicability signals used by the
    gating layer. One LLM call per retrieval; results are cached within an
    episode by (role, query_hash, candidate_id_set) to avoid redundant calls.
    """

    def __init__(
        self,
        node_name: str = "memory_critic",
        fallback_confidence: float = 0.5,
        template_name: str = "memory_critic.j2",
    ) -> None:
        self._llm = None
        self._node_name = node_name
        self._fallback_confidence = fallback_confidence
        self._template_name = template_name
        self._cache: dict[str, CriticResult] = {}

    def reset_cache(self) -> None:
        self._cache.clear()

    def _get_llm(self):
        if self._llm is None:
            self._llm = get_llm(node_name=self._node_name)
        return self._llm

    def _render_prompt(
        self,
        role: str,
        query_text: str,
        state_features: dict[str, Any],
        candidates: Sequence[MemoryCard],
    ) -> str:
        state_json = json.dumps(state_features or {}, ensure_ascii=False, default=str)
        return load_prompt(
            self._template_name,
            role=role,
            query_text=query_text,
            state_features=state_features,
            state_features_json=state_json,
            candidates=candidates,
        )

    def critique(
        self,
        role: str,
        query_text: str,
        state_features: dict[str, Any],
        candidates: Sequence[MemoryCard],
    ) -> CriticResult:
        if not candidates:
            return CriticResult(
                ranked_memory_ids=[],
                applicability_scores={},
                critic_confidence=0.0,
                recommended_action="skip",
                rationale="no candidates",
            )

        cache_key = _episode_cache_key(
            role, query_text, [c.memory_id for c in candidates]
        )
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = self._render_prompt(role, query_text, state_features, candidates)
        candidate_ids = {c.memory_id for c in candidates}

        try:
            response = self._get_llm().invoke(
                [
                    SystemMessage(
                        content="You are a precise JSON-only applicability critic."
                    ),
                    HumanMessage(content=prompt),
                ]
            )
            raw = response.content if hasattr(response, "content") else str(response)
            tokens = _extract_token_usage(response)
        except Exception:
            result = _fallback_result(candidates, self._fallback_confidence)
            self._cache[cache_key] = result
            return result

        try:
            payload = extract_json_from_text(raw)
        except Exception:
            result = _fallback_result(candidates, self._fallback_confidence)
            result.tokens = tokens
            self._cache[cache_key] = result
            return result

        if not isinstance(payload, dict):
            result = _fallback_result(candidates, self._fallback_confidence)
            result.tokens = tokens
            self._cache[cache_key] = result
            return result

        raw_scores = payload.get("per_card_applicability") or {}
        applicability: dict[str, float] = {}
        if isinstance(raw_scores, dict):
            for mid, val in raw_scores.items():
                if mid in candidate_ids:
                    applicability[mid] = _coerce_float(val, default=0.0)

        raw_rank = payload.get("ranked_memory_ids") or []
        ranked: list[str] = []
        seen: set[str] = set()
        if isinstance(raw_rank, list):
            for mid in raw_rank:
                if isinstance(mid, str) and mid in candidate_ids and mid not in seen:
                    ranked.append(mid)
                    seen.add(mid)
        for card in candidates:
            if card.memory_id not in seen:
                ranked.append(card.memory_id)
                seen.add(card.memory_id)

        for mid in ranked:
            applicability.setdefault(mid, 0.0)

        result = CriticResult(
            ranked_memory_ids=ranked,
            applicability_scores=applicability,
            critic_confidence=_coerce_float(
                payload.get("critic_confidence"), default=self._fallback_confidence
            ),
            recommended_action=_coerce_action(payload.get("recommended_action")),
            rationale=str(payload.get("rationale", "")).strip(),
            parse_failed=False,
            tokens=tokens,
        )
        self._cache[cache_key] = result
        return result


def _extract_token_usage(response: Any) -> dict[str, int] | None:
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict):
        return {
            "input_tokens": int(usage.get("input_tokens", 0) or 0),
            "output_tokens": int(usage.get("output_tokens", 0) or 0),
            "total_tokens": int(
                usage.get("total_tokens", 0)
                or usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                or 0
            ),
        }
    meta = getattr(response, "response_metadata", None) or {}
    token_usage = meta.get("token_usage") or meta.get("usage") or {}
    if token_usage:
        return {
            "input_tokens": int(
                token_usage.get("prompt_tokens") or token_usage.get("input_tokens") or 0
            ),
            "output_tokens": int(
                token_usage.get("completion_tokens") or token_usage.get("output_tokens") or 0
            ),
            "total_tokens": int(token_usage.get("total_tokens") or 0),
        }
    return None
