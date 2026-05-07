from __future__ import annotations
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping


def _key_tuple(row: Mapping, key: tuple[str, ...]) -> tuple:
    return tuple(row.get(k) for k in key)


def append_jsonl_unique(path: Path, row: Mapping, *, key: tuple[str, ...]) -> bool:
    """Append `row` to `path` (JSONL) unless its key tuple already exists.

    Returns True if a new row was appended, False if it was a duplicate.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    seen = load_done_keys(path, key=key)
    k = _key_tuple(row, key)
    if k in seen:
        return False
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    return True


def load_done_keys(path: Path, *, key: tuple[str, ...]) -> set[tuple]:
    path = Path(path)
    if not path.exists():
        return set()
    out: set[tuple] = set()
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.add(_key_tuple(row, key))
    return out


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "unknown"


def write_run_manifest(
    path: Path,
    *,
    stage: str,
    endpoints: Mapping[str, str] | None = None,
    counts: Mapping[str, int] | None = None,
    extra: Mapping | None = None,
) -> None:
    payload = {
        "stage": stage,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "endpoints": dict(endpoints or {}),
        "counts": dict(counts or {}),
    }
    if extra:
        payload["extra"] = dict(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
