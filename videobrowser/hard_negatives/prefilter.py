from __future__ import annotations
from typing import Optional

MAX_DURATION_SECONDS = 3600          # 1 hour
MAX_FILESIZE_BYTES = 1 * 1024**3     # 1 GB


def drop_reason(
    *,
    duration_seconds: Optional[int],
    filesize_bytes: Optional[int],
    video_id: str,
    pool_ids: set[str],
    gt_video_ids: set[str],
) -> Optional[str]:
    """Return a stable drop-reason string, or None if the candidate is kept.

    Unknown duration or filesize is NOT a drop reason here — Stage 4 re-checks
    the bound after download.
    """
    if video_id in gt_video_ids:
        return "gt_video"
    if video_id in pool_ids:
        return "already_in_pool"
    if duration_seconds is not None and duration_seconds > MAX_DURATION_SECONDS:
        return "duration_over_limit"
    if filesize_bytes is not None and filesize_bytes > MAX_FILESIZE_BYTES:
        return "filesize_over_limit"
    return None
