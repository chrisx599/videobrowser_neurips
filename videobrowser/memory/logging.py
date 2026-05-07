from __future__ import annotations

from pathlib import Path

from videobrowser.memory.schemas import ExperienceEvent


def append_event(path: str | Path, event: ExperienceEvent) -> None:
    event_path = Path(path)
    event_path.parent.mkdir(parents=True, exist_ok=True)
    with event_path.open("a", encoding="utf-8") as handle:
        handle.write(event.model_dump_json())
        handle.write("\n")
