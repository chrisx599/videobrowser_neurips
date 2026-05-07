from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np

from videobrowser.memory.critic import CriticResult, MemoryCritic
from videobrowser.memory.index import MemoryIndex
from videobrowser.memory.schemas import MemoryCard, RetrievalContext
from videobrowser.memory.uncertainty import compute_retrieval_uncertainty


RankerSource = Literal["bandit", "llm_critic", "cosine_fallback", "cosine", "translation_hybrid"]
RecommendedAction = Literal["apply", "skip", "loop"]


@dataclass
class RetrievalResult:
    candidate_cards: list[MemoryCard]
    selected_cards: list[MemoryCard]
    selector_scores: dict[str, float]
    selected_feature_vectors: dict[str, list[float]]
    query_embedding: np.ndarray | None = None
    selected_memory_embeddings: dict[str, np.ndarray] | None = None
    applicability_scores: dict[str, float] | None = None
    critic_confidence: float | None = None
    critic_rationale: str | None = None
    retrieval_uncertainty: float | None = None
    uncertainty_components: dict[str, float] | None = None
    combined_confidence: float | None = None
    recommended_action: RecommendedAction | None = None
    ranker_source: RankerSource | None = None


class MemoryRetriever:
    def __init__(
        self,
        store,
        index: MemoryIndex,
        embedder,
        selector,
        critic: MemoryCritic | None = None,
        scorer: Any = None,
    ):
        self.store = store
        self.index = index
        self.embedder = embedder
        self.selector = selector
        self._critic = critic
        self.scorer = scorer  # optional UtilityScorer; blended into applicability when present

    @property
    def critic(self) -> MemoryCritic:
        if self._critic is None:
            self._critic = MemoryCritic()
        return self._critic

    def reset_critic_cache(self) -> None:
        if self._critic is not None:
            self._critic.reset_cache()

    def _build_feature_map(
        self,
        candidate_cards: list[MemoryCard],
        proposal_scores: dict[str, float],
    ) -> dict[str, np.ndarray]:
        feature_map: dict[str, np.ndarray] = {}
        for card in candidate_cards:
            feature_map[card.memory_id] = np.array(
                [
                    proposal_scores.get(card.memory_id, 0.0),
                    1.0 if card.outcome == "success" else 0.0,
                    float(len(card.gap_tags)),
                ],
                dtype=float,
            )
        return feature_map

    def _is_neural_ucb(self) -> bool:
        return hasattr(self.selector, "emb_dim")

    def retrieve(self, context: RetrievalContext) -> RetrievalResult:
        filter_tags = context.filter_tags or {}
        candidate_pool = self.store.filter_cards(
            role=context.role,
            phase_tag=filter_tags.get("phase_tag"),
            gap_tags=filter_tags.get("gap_tags"),
            include_failure_memories=filter_tags.get("include_failure_memories", True),
        )
        if not candidate_pool:
            return RetrievalResult(
                candidate_cards=[],
                selected_cards=[],
                selector_scores={},
                selected_feature_vectors={},
            )
        candidate_by_id = {card.memory_id: card for card in candidate_pool}
        query_embedding = self.embedder.embed_query(context.query_text)
        proposals = self.index.top_n(query_embedding, context.proposal_top_n)

        candidate_cards: list[MemoryCard] = []
        proposal_scores: dict[str, float] = {}
        for memory_id, score in proposals:
            card = candidate_by_id.get(memory_id)
            if card is None:
                continue
            candidate_cards.append(card)
            proposal_scores[memory_id] = score

        if self._is_neural_ucb():
            memory_embeddings = {}
            for card in candidate_cards:
                try:
                    memory_embeddings[card.memory_id] = self.index.get_embedding(card.memory_id)
                except KeyError:
                    continue
            ranked = self.selector.rank(query_embedding, memory_embeddings) if memory_embeddings else []
        else:
            feature_map = self._build_feature_map(candidate_cards, proposal_scores)
            ranked = self.selector.rank(feature_map) if feature_map else []

        selected_cards = [
            candidate_by_id[memory_id]
            for memory_id, _ in ranked[: context.selection_top_k]
            if memory_id in candidate_by_id
        ]

        if self._is_neural_ucb():
            selected_feature_vectors = {}
            selected_mem_embs = {
                card.memory_id: memory_embeddings[card.memory_id]
                for card in selected_cards
                if card.memory_id in memory_embeddings
            }
        else:
            selected_feature_vectors = {
                memory_id: feature_map[memory_id].tolist()
                for memory_id, _ in ranked[: context.selection_top_k]
                if memory_id in feature_map
            }
            selected_mem_embs = None

        return RetrievalResult(
            candidate_cards=candidate_cards,
            selected_cards=selected_cards,
            selector_scores={memory_id: score for memory_id, score in ranked},
            selected_feature_vectors=selected_feature_vectors,
            query_embedding=query_embedding,
            selected_memory_embeddings=selected_mem_embs,
            ranker_source="bandit",
        )

    def retrieve_with_meta(self, context: RetrievalContext) -> RetrievalResult:
        """Retrieval with optional LLM-reranker + applicability gating.

        Dispatches on `config.memory.ranker` in {"cosine", "bandit", "llm_critic"}.
        - "cosine": pure embedding cosine top-K, no bandit, no critic.
        - "bandit": cosine top-N → LinUCB / NeuralUCB re-rank → top-K. Critic
          optionally attached for uncertainty/routing only.
        - "llm_critic": cosine top-N → LLM critic reranks AND scores.
        """
        from videobrowser.config import get_config

        config = get_config()
        ranker = getattr(config.memory, "ranker", "bandit")
        critic_cfg = getattr(config.memory, "critic", None)
        uncertainty_cfg = getattr(config.memory, "uncertainty", None)

        if ranker == "cosine":
            return self._retrieve_cosine_only(context)

        if ranker == "bandit":
            result = self.retrieve(context)
            if critic_cfg is None or not critic_cfg.enabled:
                return result
            self._attach_uncertainty_and_routing(result, uncertainty_cfg, critic_cfg)
            return result

        return self._retrieve_with_llm_critic(context, critic_cfg, uncertainty_cfg)

    def _retrieve_cosine_only(self, context: RetrievalContext) -> RetrievalResult:
        """Pure cosine top-K against the index — no bandit, no critic.

        Filters by role/phase/gap_tags, embeds the query once, ranks against
        index_text embeddings, and returns the top `selection_top_k` cards in
        cosine order. selector_scores carry the cosine similarities.
        """
        filter_tags = context.filter_tags or {}
        candidate_pool = self.store.filter_cards(
            role=context.role,
            phase_tag=filter_tags.get("phase_tag"),
            gap_tags=filter_tags.get("gap_tags"),
            include_failure_memories=filter_tags.get("include_failure_memories", True),
        )
        if not candidate_pool:
            return RetrievalResult(
                candidate_cards=[],
                selected_cards=[],
                selector_scores={},
                selected_feature_vectors={},
                ranker_source="cosine",
            )

        candidate_by_id = {card.memory_id: card for card in candidate_pool}
        query_embedding = self.embedder.embed_query(context.query_text)
        proposals = self.index.top_n(query_embedding, context.proposal_top_n)

        candidate_cards: list[MemoryCard] = []
        proposal_scores: dict[str, float] = {}
        for memory_id, score in proposals:
            card = candidate_by_id.get(memory_id)
            if card is None:
                continue
            candidate_cards.append(card)
            proposal_scores[memory_id] = float(score)

        selected_cards = candidate_cards[: context.selection_top_k]
        selected_memory_embeddings: dict[str, np.ndarray] = {}
        for card in selected_cards:
            try:
                selected_memory_embeddings[card.memory_id] = self.index.get_embedding(card.memory_id)
            except KeyError:
                continue

        return RetrievalResult(
            candidate_cards=candidate_cards,
            selected_cards=selected_cards,
            selector_scores=proposal_scores,
            selected_feature_vectors={
                card.memory_id: [proposal_scores.get(card.memory_id, 0.0)]
                for card in selected_cards
            },
            query_embedding=query_embedding,
            selected_memory_embeddings=selected_memory_embeddings or None,
            ranker_source="cosine",
        )

    def _retrieve_with_llm_critic(
        self,
        context: RetrievalContext,
        critic_cfg: Any,
        uncertainty_cfg: Any,
    ) -> RetrievalResult:
        filter_tags = context.filter_tags or {}
        candidate_pool = self.store.filter_cards(
            role=context.role,
            phase_tag=filter_tags.get("phase_tag"),
            gap_tags=filter_tags.get("gap_tags"),
            include_failure_memories=filter_tags.get("include_failure_memories", True),
        )
        if not candidate_pool:
            return RetrievalResult(
                candidate_cards=[],
                selected_cards=[],
                selector_scores={},
                selected_feature_vectors={},
                ranker_source="llm_critic",
            )

        candidate_by_id = {card.memory_id: card for card in candidate_pool}
        query_embedding = self.embedder.embed_query(context.query_text)
        proposals = self.index.top_n(query_embedding, context.proposal_top_n)

        candidate_cards: list[MemoryCard] = []
        proposal_scores: dict[str, float] = {}
        for memory_id, score in proposals:
            card = candidate_by_id.get(memory_id)
            if card is None:
                continue
            candidate_cards.append(card)
            proposal_scores[memory_id] = score

        if not candidate_cards:
            return RetrievalResult(
                candidate_cards=[],
                selected_cards=[],
                selector_scores={},
                selected_feature_vectors={},
                query_embedding=query_embedding,
                ranker_source="llm_critic",
            )

        critic_result: CriticResult = self.critic.critique(
            role=context.role,
            query_text=context.query_text,
            state_features=context.state_features or {},
            candidates=candidate_cards,
        )

        # Optional hybrid scoring: blend critic applicability with learned scorer
        from videobrowser.config import get_config as _get_cfg
        _cfg = _get_cfg()
        scorer_cfg = getattr(_cfg.memory, "scorer", None)
        applicability_map = dict(critic_result.applicability_scores)

        if (
            self.scorer is not None
            and scorer_cfg is not None
            and getattr(scorer_cfg, "enabled", False)
        ):
            alpha = float(getattr(scorer_cfg, "hybrid_alpha", 0.5))
            try:
                scorer_scores = self._score_candidates_with_mlp(
                    query_embedding=query_embedding,
                    candidate_cards=candidate_cards,
                    state_features=context.state_features or {},
                )
                for mid, s in scorer_scores.items():
                    critic_s = float(applicability_map.get(mid, 0.5))
                    applicability_map[mid] = alpha * critic_s + (1.0 - alpha) * float(s)
                # Re-rank by blended applicability (descending)
                ranked_ids = [mid for mid, _ in sorted(
                    applicability_map.items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                ) if mid in candidate_by_id]
            except Exception as exc:
                # Scorer failure: fall back to critic's ranking
                import logging
                logging.getLogger(__name__).warning("scorer blend failed: %s", exc)
                ranked_ids = [mid for mid in critic_result.ranked_memory_ids if mid in candidate_by_id]
        else:
            ranked_ids = [mid for mid in critic_result.ranked_memory_ids if mid in candidate_by_id]

        for card in candidate_cards:
            if card.memory_id not in ranked_ids:
                ranked_ids.append(card.memory_id)

        selected_cards = [candidate_by_id[mid] for mid in ranked_ids[: context.selection_top_k]]

        selector_scores: dict[str, float] = {}
        for mid in ranked_ids:
            selector_scores[mid] = float(applicability_map.get(mid, 0.0))

        selected_memory_embeddings: dict[str, np.ndarray] = {}
        for card in selected_cards:
            try:
                selected_memory_embeddings[card.memory_id] = self.index.get_embedding(card.memory_id)
            except KeyError:
                continue

        ranker_source: RankerSource = "cosine_fallback" if critic_result.parse_failed else "llm_critic"

        result = RetrievalResult(
            candidate_cards=candidate_cards,
            selected_cards=selected_cards,
            selector_scores=selector_scores,
            selected_feature_vectors={
                mid: [selector_scores.get(mid, 0.0), float(proposal_scores.get(mid, 0.0))]
                for mid in [c.memory_id for c in selected_cards]
            },
            query_embedding=query_embedding,
            selected_memory_embeddings=selected_memory_embeddings or None,
            applicability_scores=dict(applicability_map),
            critic_confidence=float(critic_result.critic_confidence),
            critic_rationale=critic_result.rationale,
            ranker_source=ranker_source,
        )

        self._attach_uncertainty_and_routing(
            result,
            uncertainty_cfg,
            critic_cfg,
            baseline_recommended=critic_result.recommended_action,
        )
        return result

    def _attach_uncertainty_and_routing(
        self,
        result: RetrievalResult,
        uncertainty_cfg: Any,
        critic_cfg: Any,
        baseline_recommended: RecommendedAction | None = None,
    ) -> None:
        if uncertainty_cfg is not None and getattr(uncertainty_cfg, "enabled", False):
            embeddings_source = result.selected_memory_embeddings or {}
            embeddings = [embeddings_source[c.memory_id] for c in result.selected_cards if c.memory_id in embeddings_source]
            scores = [float(result.selector_scores.get(c.memory_id, 0.0)) for c in result.selected_cards]
            weights = (
                float(getattr(uncertainty_cfg, "score_var_weight", 0.4)),
                float(getattr(uncertainty_cfg, "outcome_entropy_weight", 0.3)),
                float(getattr(uncertainty_cfg, "embedding_spread_weight", 0.3)),
            )
            uncertainty, components = compute_retrieval_uncertainty(
                cards=result.selected_cards,
                scores=scores,
                embeddings=embeddings if embeddings else None,
                weights=weights,
            )
            result.retrieval_uncertainty = uncertainty
            result.uncertainty_components = components

        critic_conf = result.critic_confidence
        uncertainty = result.retrieval_uncertainty

        if critic_conf is None and uncertainty is None:
            combined = None
        elif critic_conf is None:
            combined = 1.0 - float(uncertainty)
        elif uncertainty is None:
            combined = float(critic_conf)
        else:
            combined = float(critic_conf) * (1.0 - float(uncertainty))
        result.combined_confidence = combined

        if critic_cfg is None or not getattr(critic_cfg, "enabled", False):
            if baseline_recommended is not None:
                result.recommended_action = baseline_recommended
            return

        low = float(getattr(critic_cfg, "low_threshold", 0.3))
        high = float(getattr(critic_cfg, "high_threshold", 0.7))

        if baseline_recommended == "loop":
            result.recommended_action = "loop"
            return

        if combined is None:
            result.recommended_action = baseline_recommended or "apply"
            return

        if combined >= high:
            result.recommended_action = "apply"
        elif combined < low:
            result.recommended_action = "skip"
        else:
            result.recommended_action = baseline_recommended or "loop"

    def _score_candidates_with_mlp(
        self,
        query_embedding: np.ndarray,
        candidate_cards: list[MemoryCard],
        state_features: dict,
    ) -> dict[str, float]:
        """Run the UtilityScorer MLP over candidates. Returns {memory_id: score}."""
        from videobrowser.memory.scorer import featurize_state

        if self.scorer is None or not candidate_cards:
            return {}
        state_keys = getattr(self.scorer, "state_feature_keys", None) or []
        state_vec = featurize_state(state_features, state_keys)

        card_embs: list[np.ndarray] = []
        card_ids: list[str] = []
        for card in candidate_cards:
            try:
                emb = self.index.get_embedding(card.memory_id)
            except KeyError:
                continue
            card_embs.append(np.asarray(emb, dtype=float))
            card_ids.append(card.memory_id)
        if not card_embs:
            return {}

        q = np.asarray(query_embedding, dtype=float)
        card_mat = np.stack(card_embs, axis=0)
        scores = self.scorer.score_batch(q, card_mat, state_vec)
        return {cid: float(s) for cid, s in zip(card_ids, scores)}
