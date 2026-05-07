from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from videobrowser.config import AppConfig
from videobrowser.memory.bandit import LinUCBSelector
from videobrowser.memory.neural_ucb import NeuralUCBSelector
from videobrowser.memory.embedder import FrozenEmbedder
from videobrowser.memory.index import MemoryIndex
from videobrowser.memory.retriever import MemoryRetriever
from videobrowser.memory.store import MemoryStore
from videobrowser.utils.logger import get_logger


def _maybe_load_scorer(config: AppConfig):
    """Lazy-load UtilityScorer if config.memory.scorer.enabled and path exists."""
    scorer_cfg = getattr(config.memory, "scorer", None)
    if scorer_cfg is None or not getattr(scorer_cfg, "enabled", False):
        return None
    model_path = getattr(scorer_cfg, "model_path", None)
    if not model_path:
        return None
    model_p = Path(model_path)
    if not model_p.exists():
        get_logger().log(
            "Memory", "scorer_path_missing", {"path": str(model_p)}, level="WARN"
        )
        return None
    try:
        from videobrowser.memory.scorer import UtilityScorer
        scorer = UtilityScorer.load(model_p)
        get_logger().log("Memory", "scorer_loaded", {"path": str(model_p)}, level="INFO")
        return scorer
    except Exception as exc:
        get_logger().log(
            "Memory", "scorer_load_failed", {"path": str(model_p), "error": str(exc)}, level="WARN"
        )
        return None


def _build_embedder(config: AppConfig) -> FrozenEmbedder:
    """Prefer the HTTP-backed BGE-M3 vLLM server when BGE_M3_EMBEDDING_URL is
    set — avoids per-worker SentenceTransformer copies under multi-worker eval.
    Falls back to loading SentenceTransformer locally otherwise.
    """
    import os

    url = os.environ.get("BGE_M3_EMBEDDING_URL")
    if url:
        try:
            from experiments.local_inference.http_embedder import HttpEmbeddingBackend

            model = os.environ.get("BGE_M3_EMBEDDING_MODEL", "BAAI/bge-m3")
            batch = int(os.environ.get("BGE_M3_EMBEDDING_BATCH", "64"))
            query_prefix = os.environ.get("BGE_M3_EMBEDDING_QUERY_PREFIX", "")
            backend = HttpEmbeddingBackend(
                url=url, model=model, batch_size=batch, query_prefix=query_prefix,
            )
            get_logger().log(
                "Memory", "embedder_http", {"url": url, "model": model}, level="INFO"
            )
            return FrozenEmbedder(backend)
        except Exception as exc:
            get_logger().log(
                "Memory",
                "embedder_http_failed",
                {"url": url, "error": str(exc)},
                level="WARN",
            )

    from sentence_transformers import SentenceTransformer

    return FrozenEmbedder(SentenceTransformer(config.memory.embedding.model_name))


@dataclass
class MemoryRuntime:
    retriever: MemoryRetriever | None
    unavailable_reason: str | None = None


class _LazyEmbedderBackend:
    """Defers construction of the embedding backend until first use.

    This lets the translation_hybrid bootstrap path succeed even when the
    underlying SentenceTransformer model is not yet cached locally — the model
    is only loaded the first time an encoding call is made (i.e. during
    actual retrieval, not during bootstrap).
    """

    def __init__(self, config: AppConfig) -> None:
        self._config = config
        self._backend = None

    def _materialize(self):
        if self._backend is None:
            self._backend = _build_embedder(self._config).backend
        return self._backend

    def encode(self, texts, **kwargs):
        return self._materialize().encode(texts, **kwargs)


_DEFAULT_RUNTIME_CACHE_KEY: tuple[object, ...] | None = None
_DEFAULT_RUNTIME: MemoryRuntime | None = None


def _runtime_cache_key(config: AppConfig) -> tuple[object, ...]:
    if not config.memory.enabled:
        return (False,)
    return (
        bool(config.memory.enabled),
        config.memory.store_path,
        config.memory.embedding_path,
        config.memory.embedding.model_name,
        config.memory.bandit.alpha,
    )


def build_memory_runtime(config: AppConfig) -> MemoryRuntime:
    if not config.memory.enabled:
        return MemoryRuntime(retriever=None)

    logger = get_logger()
    ranker = getattr(config.memory, "ranker", "bandit")

    try:
        if not Path(config.memory.embedding_path).exists():
            raise FileNotFoundError(config.memory.embedding_path)

        store = MemoryStore(config.memory.store_path)

        if ranker == "translation_hybrid":
            from videobrowser.memory.translation_index import TranslationIndex
            from videobrowser.memory.translation_retriever import (
                TranslationMemoryRetriever,
                ZQExtractor,
            )
            translation_cfg = config.memory.translation
            t_index = TranslationIndex.load(config.memory.embedding_path)
            # Use a lazy backend so bootstrap succeeds even when the embedding
            # model is not yet cached locally; the backend is only materialised
            # on the first actual encode() / embed_texts() call.
            embedder_backend = _LazyEmbedderBackend(config)

            # Build the z_Q LLM callable from translation.zq_llm
            from openai import OpenAI as _OpenAI
            client = _OpenAI(base_url=translation_cfg.zq_llm.base_url, api_key="EMPTY")

            def _zq_llm(prompt: str) -> str:
                resp = client.chat.completions.create(
                    model=translation_cfg.zq_llm.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=translation_cfg.zq_llm.temperature,
                    max_tokens=translation_cfg.zq_llm.max_tokens,
                )
                return (resp.choices[0].message.content or "").strip()

            zq_extractor = ZQExtractor(llm=_zq_llm, cache_path=translation_cfg.zq_cache_path)
            retriever = TranslationMemoryRetriever(
                store=store,
                index=t_index,
                embedder=embedder_backend,  # lazy wrapper; materialises SentenceTransformer / HTTP backend on first use
                zq_extractor=zq_extractor,
                lambda_q=translation_cfg.lambda_q,
                lambda_z=translation_cfg.lambda_z,
                lambda_t=translation_cfg.lambda_t,
            )
            return MemoryRuntime(retriever=retriever)

        # Existing path (bandit / llm_critic / cosine)
        index = MemoryIndex.load(config.memory.embedding_path)
        embedder = _build_embedder(config)
        bandit_cfg = config.memory.bandit
        algorithm = getattr(bandit_cfg, "algorithm", "linucb")

        if algorithm == "neural_ucb":
            model_path = Path(getattr(bandit_cfg, "model_path", ""))
            meta_path = Path(getattr(bandit_cfg, "meta_path", ""))
            if model_path.exists() and meta_path.exists():
                selector = NeuralUCBSelector.load(model_path, meta_path)
            else:
                selector = NeuralUCBSelector(
                    emb_dim=getattr(bandit_cfg, "emb_dim", 1024),
                    hidden_dim=getattr(bandit_cfg, "hidden_dim", 128),
                    output_dim=getattr(bandit_cfg, "output_dim", 64),
                    alpha=bandit_cfg.alpha,
                )
        else:
            bandit_state_path = Path(config.memory.bandit_state_path)
            if bandit_state_path.exists():
                selector = LinUCBSelector.load(bandit_state_path)
            else:
                selector = LinUCBSelector(alpha=bandit_cfg.alpha, feature_dim=3)
        scorer = _maybe_load_scorer(config)
        retriever = MemoryRetriever(
            store=store, index=index, embedder=embedder, selector=selector, scorer=scorer,
        )
        return MemoryRuntime(retriever=retriever)
    except Exception as exc:
        reason = f"memory runtime unavailable: {exc}"
        logger.log(
            "Memory",
            "runtime_unavailable",
            {"reason": reason},
            level="WARN",
        )
        return MemoryRuntime(retriever=None, unavailable_reason=reason)


def get_default_memory_runtime(config: AppConfig) -> MemoryRuntime:
    global _DEFAULT_RUNTIME, _DEFAULT_RUNTIME_CACHE_KEY

    cache_key = _runtime_cache_key(config)
    if _DEFAULT_RUNTIME is None or _DEFAULT_RUNTIME_CACHE_KEY != cache_key:
        _DEFAULT_RUNTIME = build_memory_runtime(config)
        _DEFAULT_RUNTIME_CACHE_KEY = cache_key
    return _DEFAULT_RUNTIME


def reset_default_memory_runtime() -> None:
    global _DEFAULT_RUNTIME, _DEFAULT_RUNTIME_CACHE_KEY
    _DEFAULT_RUNTIME = None
    _DEFAULT_RUNTIME_CACHE_KEY = None
    reset_default_skill_bank()


_DEFAULT_SKILL_BANK = None
_DEFAULT_SKILL_BANK_KEY: tuple | None = None


def get_default_skill_bank(config: AppConfig):
    """Lazy-load SkillBank. Shared with the episodic-memory path only via
    the same `_build_embedder` factory."""
    global _DEFAULT_SKILL_BANK, _DEFAULT_SKILL_BANK_KEY
    from videobrowser.memory.skill_bank import SkillBank

    key = (
        config.memory.skill_bank_path,
        config.memory.enabled,
        config.memory.source,
    )
    if _DEFAULT_SKILL_BANK is None or _DEFAULT_SKILL_BANK_KEY != key:
        embedder = _build_embedder(config)
        _DEFAULT_SKILL_BANK = SkillBank.load(
            config.memory.skill_bank_path, embedder=embedder
        )
        _DEFAULT_SKILL_BANK_KEY = key
    return _DEFAULT_SKILL_BANK


def reset_default_skill_bank() -> None:
    global _DEFAULT_SKILL_BANK, _DEFAULT_SKILL_BANK_KEY
    _DEFAULT_SKILL_BANK = None
    _DEFAULT_SKILL_BANK_KEY = None
