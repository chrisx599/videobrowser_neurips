from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from videobrowser.memory.schemas import MemoryCard


class MemoryStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def load_cards(self) -> list[MemoryCard]:
        if not self.path.exists():
            return []

        cards: list[MemoryCard] = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if hasattr(MemoryCard, "model_validate"):
                    cards.append(MemoryCard.model_validate(payload))
                else:
                    cards.append(MemoryCard.parse_obj(payload))
        return cards

    def write_cards(self, cards: Iterable[MemoryCard]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            for card in cards:
                if hasattr(card, "model_dump"):
                    payload = card.model_dump(mode="json")
                else:
                    payload = card.dict()
                handle.write(json.dumps(payload, ensure_ascii=True))
                handle.write("\n")

    def append_card(self, card: MemoryCard) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if hasattr(card, "model_dump"):
            payload = card.model_dump(mode="json")
        else:
            payload = card.dict()
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=True))
            handle.write("\n")

    def filter_cards(
        self,
        role: str,
        phase_tag: str | None = None,
        gap_tags: list[str] | None = None,
        include_failure_memories: bool = True,
    ) -> list[MemoryCard]:
        cards = self.load_cards()
        filtered: list[MemoryCard] = []

        for card in cards:
            if card.role != role:
                continue
            if not include_failure_memories and card.outcome == "failure":
                continue
            filtered.append(card)

        return filtered
