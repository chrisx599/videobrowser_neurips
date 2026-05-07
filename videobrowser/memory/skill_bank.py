from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Protocol

import numpy as np

from videobrowser.memory.skill_schema import VideoSearchSkill


class EmbedderLike(Protocol):
    """Matches FrozenEmbedder (videobrowser/memory/embedder.py) API."""
    def embed_texts(self, texts: List[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


@dataclass
class RetrievedSkill:
    skill: VideoSearchSkill
    score: float


class SkillBank:
    def __init__(self, skills: List[VideoSearchSkill], embeddings: np.ndarray):
        self.skills = skills
        self.embeddings = embeddings  # shape (n_skills, dim)

    @classmethod
    def load(cls, path: Path | str, *, embedder: EmbedderLike) -> "SkillBank":
        p = Path(path)
        skills: List[VideoSearchSkill] = []
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                skills.append(VideoSearchSkill(**json.loads(line)))

        if not skills:
            return cls([], np.zeros((0, 0), dtype=np.float32))

        texts = [f"{s.trigger_condition}\n{s.procedure}" for s in skills]
        embs = np.asarray(embedder.embed_texts(texts), dtype=np.float32)
        return cls(skills, embs)

    def retrieve(
        self,
        query: str,
        *,
        top_k: int = 3,
        similarity_floor: float = 0.0,
        embedder: Optional[EmbedderLike] = None,
    ) -> List[RetrievedSkill]:
        if not self.skills:
            return []
        if embedder is None:
            raise ValueError(
                "SkillBank.retrieve requires embedder= (caller-provided, "
                "so we don't keep one per bank and break multiprocessing)."
            )
        q_emb = np.asarray(embedder.embed_query(query), dtype=np.float32)
        # normalized dot-product == cosine
        scores = self.embeddings @ q_emb
        idxs = np.argsort(-scores)[: top_k]
        out: List[RetrievedSkill] = []
        for i in idxs:
            score = float(scores[i])
            if score < similarity_floor:
                continue
            out.append(RetrievedSkill(skill=self.skills[int(i)], score=score))
        return out
