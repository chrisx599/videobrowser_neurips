# videobrowser/memory/translation_retriever.py
"""v2c TranslationMemoryRetriever — hybrid scoring + cached z_Q extraction.

The extractor is a thin wrapper around an LLM callable plus a JSONL cache
keyed by sha256(question.strip()). Corrupt or duplicate lines are skipped.

The full retriever (hybrid scoring + RetrievalResult assembly) is added in
Task 9.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from jinja2 import Template
from pydantic import ValidationError

from videobrowser.memory.translation_schemas import ZSignature, parse_z_signature


_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.MULTILINE)
_PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "memory_v2c_zq_extract.j2"
_PROMPT_CACHE: dict = {}


def _load_prompt() -> "Template":
    if _PROMPT_CACHE.get("t") is None:
        _PROMPT_CACHE["t"] = Template(_PROMPT_PATH.read_text(encoding="utf-8"))
    return _PROMPT_CACHE["t"]


def _hash_question(question: str) -> str:
    return hashlib.sha256(question.strip().encode("utf-8")).hexdigest()


def _parse_zq(raw: str) -> ZSignature:
    body = _FENCE_RE.sub("", raw or "").strip()
    if not body:
        raise ValueError("empty z_Q response")
    try:
        obj = json.loads(body)
    except json.JSONDecodeError as exc:
        raise ValueError(f"z_Q response is not valid JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ValueError("z_Q response root must be a JSON object")
    try:
        return parse_z_signature(obj)
    except ValidationError as exc:
        raise ValueError(f"z_Q schema invalid: {exc}") from exc


@dataclass
class _CacheEntry:
    q_hash: str
    z_q: ZSignature


class ZQExtractor:
    """One-LLM-call-per-unique-question extractor with a JSONL cache.

    The cache file format:
        {"q_hash": "...", "question_preview": "...", "z_q": {...}}

    Cache is loaded eagerly from `cache_path` at construction. Subsequent
    extracts that miss the cache append a new line to the file.
    """

    def __init__(self, *, llm: Callable[[str], str], cache_path: str | Path) -> None:
        self._llm = llm
        self._cache_path = Path(cache_path)
        self._cache: dict[str, ZSignature] = self._load_cache()

    def _load_cache(self) -> dict[str, ZSignature]:
        if not self._cache_path.exists():
            return {}
        out: dict[str, ZSignature] = {}
        for line in self._cache_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                z_q = ZSignature(**obj["z_q"])
                out[obj["q_hash"]] = z_q
            except Exception:
                # Skip corrupt lines without aborting load.
                continue
        return out

    def _append_cache(self, q_hash: str, question: str, z_q: ZSignature) -> None:
        self._cache_path.parent.mkdir(parents=True, exist_ok=True)
        with self._cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "q_hash": q_hash,
                "question_preview": question[:140],
                "z_q": z_q.model_dump(mode="json"),
            }) + "\n")

    def extract(self, question: str) -> ZSignature:
        q_hash = _hash_question(question)
        cached = self._cache.get(q_hash)
        if cached is not None:
            return cached
        prompt = _load_prompt().render(question=question.strip())
        raw = self._llm(prompt)
        z_q = _parse_zq(raw)
        self._cache[q_hash] = z_q
        self._append_cache(q_hash, question.strip(), z_q)
        return z_q


def extract_zq(question: str, *, llm: Callable[[str], str], cache_path: str | Path) -> ZSignature:
    """Module-level convenience for one-shot extraction."""
    return ZQExtractor(llm=llm, cache_path=cache_path).extract(question)


# ---------------------------------------------------------------------------
# TranslationMemoryRetriever (Task 9)
# ---------------------------------------------------------------------------
import numpy as np

from videobrowser.memory.retriever import RetrievalResult
from videobrowser.memory.schemas import MemoryCard, RetrievalContext
from videobrowser.memory.store import MemoryStore
from videobrowser.memory.translation_index import TranslationIndex


def _z_text_query_string(z_q: ZSignature) -> str:
    parts = [z_q.visual_target, z_q.searchable_context, z_q.search_warning]
    parts = [p.strip() for p in parts if p and p.strip()]
    return " · ".join(parts) if parts else "(empty)"


def _z_type_query_string(z_q: ZSignature) -> str:
    return f"{z_q.evidence_type} · {z_q.answer_type}"


class TranslationMemoryRetriever:
    """Hybrid retriever for v2c banks. Pure cosine, no bandit, no critic.

    Composition:
      - `store`: MemoryStore (filter cards by role / phase / gap_tags).
      - `index`: TranslationIndex (three-matrix npz).
      - `embedder`: SentenceTransformer or HttpEmbeddingBackend (any object
        with `.encode(texts) -> ndarray` or `.embed_texts(texts) -> ndarray`).
      - `zq_extractor`: ZQExtractor (LLM + cache).
      - `lambda_q`, `lambda_z`, `lambda_t`: hybrid weights.
    """

    def __init__(
        self,
        *,
        store: MemoryStore,
        index: TranslationIndex,
        embedder,
        zq_extractor: ZQExtractor,
        lambda_q: float,
        lambda_z: float,
        lambda_t: float,
        _embedder_text_keys: tuple[str, str, str] | None = None,
    ) -> None:
        self.store = store
        self.index = index
        self.embedder = embedder
        self.zq_extractor = zq_extractor
        self.lambda_q = lambda_q
        self.lambda_z = lambda_z
        self.lambda_t = lambda_t
        # Test-only injection point: lets a stub embedder label the three
        # query texts via fixed keys instead of the real strings.
        self._test_keys = _embedder_text_keys

    def _embed_one(self, text: str) -> np.ndarray:
        if hasattr(self.embedder, "embed_texts"):
            arr = self.embedder.embed_texts([text])
        else:
            arr = self.embedder.encode([text], convert_to_numpy=True)
        return np.asarray(arr, dtype=float)[0]

    def _embed_three(self, q: str, z_txt: str, z_typ: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if self._test_keys is not None:
            tq, tz, tt = self._test_keys
            arr = self.embedder.encode([tq, tz, tt])
            arr = np.asarray(arr, dtype=float)
            return arr[0], arr[1], arr[2]
        # Real path: one batched call for cache-friendliness.
        if hasattr(self.embedder, "embed_texts"):
            arr = self.embedder.embed_texts([q, z_txt, z_typ])
        else:
            arr = self.embedder.encode([q, z_txt, z_typ], convert_to_numpy=True)
        arr = np.asarray(arr, dtype=float)
        return arr[0], arr[1], arr[2]

    def retrieve_with_meta(self, context: RetrievalContext) -> RetrievalResult:
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
                ranker_source="translation_hybrid",
            )

        candidate_by_id = {card.memory_id: card for card in candidate_pool}

        z_q = self.zq_extractor.extract(context.query_text)
        emb_q, emb_z_text, emb_z_type = self._embed_three(
            context.query_text,
            _z_text_query_string(z_q),
            _z_type_query_string(z_q),
        )
        proposals = self.index.top_n(
            query_q=emb_q, query_z_text=emb_z_text, query_z_type=emb_z_type,
            n=context.proposal_top_n,
            lambda_q=self.lambda_q, lambda_z=self.lambda_z, lambda_t=self.lambda_t,
        )

        candidate_cards: list[MemoryCard] = []
        proposal_scores: dict[str, float] = {}
        for memory_id, score in proposals:
            card = candidate_by_id.get(memory_id)
            if card is None:
                continue
            candidate_cards.append(card)
            proposal_scores[memory_id] = score

        selected_cards = candidate_cards[: context.selection_top_k]

        return RetrievalResult(
            candidate_cards=candidate_cards,
            selected_cards=selected_cards,
            selector_scores=proposal_scores,
            selected_feature_vectors={},
            query_embedding=emb_q,
            ranker_source="translation_hybrid",
        )
