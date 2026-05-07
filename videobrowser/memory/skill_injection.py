from __future__ import annotations

from typing import List

from videobrowser.memory.skill_bank import RetrievedSkill


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Char-budget approximation: assume 4 chars ≈ 1 token."""
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "…"


def render_skill_block(
    retrieved: List[RetrievedSkill],
    *,
    max_injected: int = 2,
    per_skill_max_tokens: int = 80,
) -> str:
    if not retrieved:
        return ""
    kept = retrieved[: max_injected]
    lines = [
        "[Relevant Strategies]",
        "The following strategies were distilled from past similar episodes. "
        "Treat them as suggestions, not commands:",
        "",
    ]
    for i, r in enumerate(kept, 1):
        s = r.skill
        n = len(s.supporting_episode_ids)
        successful = max(0, round(s.success_rate * n))
        proc = _truncate_to_tokens(s.procedure, per_skill_max_tokens)
        trig = _truncate_to_tokens(s.trigger_condition, per_skill_max_tokens)
        lines.append(f"{i}. {s.name}: When {trig} do {proc}")
        lines.append(f"   (worked in {successful}/{n} similar past episodes)")
    lines.append("")
    return "\n".join(lines)
