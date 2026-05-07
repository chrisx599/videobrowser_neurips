from __future__ import annotations

from typing import Any

from videobrowser.memory.schemas import CheckerState


def compute_local_reward(
    before: CheckerState, after: CheckerState, runtime_flags: dict[str, Any]
) -> float:
    gap_gain = len(before.missing_slots) - len(after.missing_slots)
    verified_gain = len(after.verified_slots) - len(before.verified_slots)
    redundancy_penalty = float(bool(runtime_flags.get("repeat_query"))) + float(
        bool(runtime_flags.get("repeat_watch"))
    )
    cost_penalty = float(runtime_flags.get("cost", 0.0) or 0.0)

    return (
        (1.0 * gap_gain)
        + (0.5 * verified_gain)
        - (0.7 * redundancy_penalty)
        - (0.2 * cost_penalty)
    )
