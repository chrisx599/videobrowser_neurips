from __future__ import annotations

from typing import Optional

from videobrowser.memory.schemas import MemoryCard


def render_memory_block(
    role: str,
    cards: list[MemoryCard],
    applicability_scores: Optional[dict[str, float]] = None,
    drop_threshold: float = 0.0,
    low_confidence_nudge: bool = False,
) -> str:
    """Render selected memory cards into a prompt block.

    If `applicability_scores` is provided, cards whose score is below
    `drop_threshold` are dropped. If `low_confidence_nudge` is true, a short
    instruction is appended telling the model to treat these lessons with
    heightened skepticism — used when the critic recommends `loop`.
    """
    filtered_cards: list[MemoryCard] = []
    if applicability_scores is not None and drop_threshold > 0.0:
        for card in cards:
            score = float(applicability_scores.get(card.memory_id, 1.0))
            if score >= drop_threshold:
                filtered_cards.append(card)
    else:
        filtered_cards = list(cards)

    if not filtered_cards:
        return ""

    success_count = sum(1 for c in filtered_cards if c.outcome == "success")
    total_count = len(filtered_cards)

    lines = [
        "[Past Experience]",
        "The following lessons were learned from previous similar tasks. "
        "Use them to guide your strategy, not as hard rules:",
        "",
    ]
    for i, card in enumerate(filtered_cards, 1):
        body = card.payload_text or card.memory_text
        # When the card carries an index_text (v2b dual-text cards), surface
        # the full past question to the planner. Older cards without
        # index_text fall through unchanged.
        if card.index_text:
            text = f'Past question: "{card.index_text}"\n{body}'
        else:
            text = body
        if applicability_scores is not None:
            score = float(applicability_scores.get(card.memory_id, 1.0))
            lines.append(f"{i}. [applicability={score:.2f}] {text}")
        else:
            lines.append(f"{i}. {text}")
    lines.append("")
    lines.append(
        f"(From {total_count} past episodes: "
        f"{success_count} correct, {total_count - success_count} incorrect)"
    )
    if low_confidence_nudge:
        lines.append("")
        lines.append(
            "Note: retrieval confidence is low; treat these lessons cautiously "
            "and consider gathering more evidence before committing."
        )
    return "\n".join(lines)
