"""Build pool.jsonl for the offline search engine.

Scope (per user's directives):
  * Include ALL seed videos (no size/duration filter).
  * Include cached videos only if they already have timestamp transcripts AND
    file size <= 5 GB.
  * Transcribe any seed videos missing timestamp transcripts via whisper.
  * Fetch title/description/channel/tags via yt-dlp (proxy) into a metadata
    cache so this is restartable.
  * Assemble pool.jsonl with {id, link, title, description, channel, tags,
    upload_date, duration, transcript}.

Run sub-stages individually (idempotent):
  python scripts/build_offline_pool.py inventory
  python scripts/build_offline_pool.py transcribe [--workers 4]
  python scripts/build_offline_pool.py metadata [--workers 8]
  python scripts/build_offline_pool.py assemble
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import requests

ROOT = Path("/home/zhengyangliang/VideoBrowser")
VB_DATA = Path("/mnt/data/zhengyangliang/videobrowser")
SEED_DIR = VB_DATA / "offline_set"
CACHE_VIDEOS = VB_DATA / "data/cache/videos"
CACHE_TS = VB_DATA / "data/cache/transcripts_with_timestamps"
CACHE_AUDIO = VB_DATA / "data/cache/audio"
METADATA_CACHE = VB_DATA / "data/cache/metadata"  # one json per video_id
INVENTORY = ROOT / "data/offline_search/pool_inventory.jsonl"
POOL_JSONL = ROOT / "data/offline_search/pool.jsonl"
DATA_V45 = ROOT / "data/benchmark/videobrowsecomp/data_v4.5.jsonl"

MAX_CACHE_SIZE_BYTES = 5 * 1024 * 1024 * 1024  # 5 GB
WHISPER_URL = "http://localhost:8028/v1/audio/transcriptions"
WHISPER_MODEL = "openai/whisper-large-v3-turbo"
PROXY_PORT_POOL = list(range(8001, 8021))  # static Oxylabs ports forwarded via SSH tunnel


# ---------------------------------------------------------------------------
# Stage 1: inventory
# ---------------------------------------------------------------------------

def stage_inventory(_args: argparse.Namespace) -> None:
    INVENTORY.parent.mkdir(parents=True, exist_ok=True)

    # Gather titles from data_v4.5 (seeds are referenced with titles there).
    import ast
    titles_from_dataset: dict[str, str] = {}
    with DATA_V45.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            videos = row.get("videos")
            if isinstance(videos, str):
                try:
                    videos = ast.literal_eval(videos)
                except Exception:
                    continue
            for v in videos or []:
                if isinstance(v, dict):
                    vid = v.get("video_id")
                    title = v.get("title")
                    if vid and title:
                        titles_from_dataset.setdefault(vid, title)

    records: list[dict] = []

    # Seeds — include all, regardless of size/duration.
    for p in sorted(SEED_DIR.glob("*.mp4")):
        vid = p.stem
        records.append({
            "video_id": vid,
            "source": "seed",
            "video_path": str(p),
            "size": p.stat().st_size,
            "has_ts": (CACHE_TS / f"{vid}.json").exists() or (SEED_DIR / f"{vid}.json").exists(),
            "dataset_title": titles_from_dataset.get(vid),
        })

    seed_ids = {r["video_id"] for r in records}

    # Cached videos — require transcript, size <= 5 GB, not already a seed.
    ts_ids = {p.stem for p in CACHE_TS.glob("*.json")}
    kept, skipped_no_ts, skipped_size = 0, 0, 0
    for p in sorted(CACHE_VIDEOS.glob("*.mp4")):
        vid = p.stem
        if vid in seed_ids:
            continue
        if vid not in ts_ids:
            skipped_no_ts += 1
            continue
        sz = p.stat().st_size
        if sz > MAX_CACHE_SIZE_BYTES:
            skipped_size += 1
            continue
        records.append({
            "video_id": vid,
            "source": "cache",
            "video_path": str(p),
            "size": sz,
            "has_ts": True,
            "dataset_title": titles_from_dataset.get(vid),
        })
        kept += 1

    with INVENTORY.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    seed_total = sum(1 for r in records if r["source"] == "seed")
    seed_missing_ts = sum(1 for r in records if r["source"] == "seed" and not r["has_ts"])
    print(f"inventory written: {INVENTORY}")
    print(f"  seeds: {seed_total} ({seed_missing_ts} need transcription)")
    print(f"  cache kept: {kept} (no-ts skipped: {skipped_no_ts}, size>5GB skipped: {skipped_size})")
    print(f"  total pool candidates: {len(records)}")


# ---------------------------------------------------------------------------
# Stage 2: transcribe seed videos missing transcripts
# ---------------------------------------------------------------------------

def _extract_audio(video_path: Path, audio_path: Path) -> bool:
    if audio_path.exists() and audio_path.stat().st_size > 1024:
        return True
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
        "-i", str(video_path), "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "libmp3lame", "-q:a", "6", str(audio_path),
    ]
    r = subprocess.run(cmd, capture_output=True)
    if r.returncode != 0:
        print(f"  ffmpeg failed {video_path.name}: {r.stderr.decode()[:200]}", file=sys.stderr)
        return False
    return True


def _whisper_transcribe(audio_path: Path) -> Optional[list[dict]]:
    with audio_path.open("rb") as f:
        resp = requests.post(
            WHISPER_URL,
            files={"file": (audio_path.name, f, "audio/mpeg")},
            data={
                "model": WHISPER_MODEL,
                "response_format": "verbose_json",
                "timestamp_granularities[]": "segment",
                "language": "en",  # vllm whisper requires language
            },
            timeout=600,
        )
    if resp.status_code != 200:
        print(f"  whisper {audio_path.name} -> {resp.status_code}: {resp.text[:200]}", file=sys.stderr)
        return None
    data = resp.json()
    segs = data.get("segments") or []
    return [{"start": float(s.get("start", 0.0)), "end": float(s.get("end", 0.0)),
             "text": (s.get("text") or "").strip()}
            for s in segs if (s.get("text") or "").strip()]


def _transcribe_one(video_path: Path) -> tuple[str, str]:
    vid = video_path.stem
    out = CACHE_TS / f"{vid}.json"
    if out.exists() and out.stat().st_size > 10:
        return vid, "exists"
    audio = CACHE_AUDIO / f"{vid}.mp3"
    if not _extract_audio(video_path, audio):
        return vid, "extract_fail"
    segs = _whisper_transcribe(audio)
    if segs is None:
        return vid, "whisper_fail"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(segs, ensure_ascii=False), encoding="utf-8")
    return vid, f"ok ({len(segs)} segs)"


def stage_transcribe(args: argparse.Namespace) -> None:
    if not INVENTORY.exists():
        sys.exit(f"Run `inventory` first. {INVENTORY} missing.")
    todo: list[Path] = []
    with INVENTORY.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["source"] == "seed" and not r["has_ts"]:
                todo.append(Path(r["video_path"]))
    # double-check actual file state
    todo = [p for p in todo if not (CACHE_TS / f"{p.stem}.json").exists()]
    print(f"transcribe: {len(todo)} seed videos to process with {args.workers} workers")
    if not todo:
        return
    t0 = time.time()
    done = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        for vid, status in pool.map(_transcribe_one, todo):
            done += 1
            if done % 10 == 0 or "fail" in status:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1)
                eta = (len(todo) - done) / max(rate, 1e-6)
                print(f"  [{done}/{len(todo)}] {vid}: {status}  ({elapsed:.0f}s, ETA {eta:.0f}s)", flush=True)
    print(f"transcribe done in {time.time()-t0:.0f}s")


# ---------------------------------------------------------------------------
# Stage 3: fetch metadata via yt-dlp
# ---------------------------------------------------------------------------

def _meta_cache_path(vid: str) -> Path:
    return METADATA_CACHE / f"{vid}.json"


def _fetch_meta_one(vid: str, _worker_idx: int) -> tuple[str, str]:
    """Fetch title via YouTube oEmbed (no auth, not rate-limited, no bot
    detection). Returns just title + channel + thumbnail; that's enough for
    BM25 and matches the user's directive for cached videos."""
    out = _meta_cache_path(vid)
    if out.exists() and out.stat().st_size > 5:
        return vid, "exists"
    oembed = (
        "https://www.youtube.com/oembed?url="
        f"https://www.youtube.com/watch?v={vid}&format=json"
    )
    try:
        r = requests.get(oembed, timeout=15)
    except requests.RequestException as e:
        return vid, f"net_fail({type(e).__name__})"
    if r.status_code == 404:
        return vid, "not_found"
    if r.status_code == 401:
        # private/removed video
        return vid, "unavailable"
    if r.status_code != 200:
        return vid, f"http_{r.status_code}"
    try:
        info = r.json()
    except Exception:
        return vid, "parse_fail"
    slim = {
        "id": vid,
        "title": info.get("title"),
        "description": "",
        "channel": info.get("author_name") or "",
        "upload_date": "",
        "duration": None,
        "tags": [],
        "thumbnail": info.get("thumbnail_url") or "",
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(slim, ensure_ascii=False), encoding="utf-8")
    return vid, "ok"


def stage_metadata(args: argparse.Namespace) -> None:
    if not INVENTORY.exists():
        sys.exit(f"Run `inventory` first. {INVENTORY} missing.")
    todo: list[str] = []
    with INVENTORY.open(encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            vid = r["video_id"]
            if not _meta_cache_path(vid).exists():
                todo.append(vid)
    print(f"metadata: {len(todo)} to fetch with {args.workers} workers")
    if not todo:
        return
    t0 = time.time()
    ok = fail = 0
    with cf.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_fetch_meta_one, vid, i): vid for i, vid in enumerate(todo)}
        done = 0
        for fut in cf.as_completed(futures):
            vid, status = fut.result()
            done += 1
            if status == "ok":
                ok += 1
            elif status != "exists":
                fail += 1
            if done % 50 == 0 or "fail" in status:
                elapsed = time.time() - t0
                rate = done / max(elapsed, 1)
                eta = (len(todo) - done) / max(rate, 1e-6)
                print(f"  [{done}/{len(todo)}] {vid}: {status}  (ok={ok} fail={fail}, ETA {eta:.0f}s)", flush=True)
    print(f"metadata done in {time.time()-t0:.0f}s (ok={ok} fail={fail})")


# ---------------------------------------------------------------------------
# Stage 4: assemble pool.jsonl
# ---------------------------------------------------------------------------

def stage_assemble(_args: argparse.Namespace) -> None:
    if not INVENTORY.exists():
        sys.exit(f"Run `inventory` first. {INVENTORY} missing.")
    POOL_JSONL.parent.mkdir(parents=True, exist_ok=True)
    kept = skipped_empty_ts = 0
    no_meta = 0
    with INVENTORY.open(encoding="utf-8") as fin, POOL_JSONL.open("w", encoding="utf-8") as fout:
        for line in fin:
            r = json.loads(line)
            vid = r["video_id"]
            is_seed = r.get("source") == "seed"
            ts_path = CACHE_TS / f"{vid}.json"
            segs: list = []
            if ts_path.exists():
                try:
                    segs = json.loads(ts_path.read_text(encoding="utf-8"))
                except Exception:
                    segs = []
            text_total = sum(len((s.get("text") or "").strip()) for s in segs if isinstance(s, dict))
            # Seeds are always kept (they are ground-truth answer videos and
            # can still be matched via title/channel even with empty transcripts).
            # Cached distractors with <20 chars of transcript are dropped to
            # avoid indexing noise.
            if not is_seed and text_total < 20:
                skipped_empty_ts += 1
                continue

            meta_path = _meta_cache_path(vid)
            meta = {}
            if meta_path.exists():
                try:
                    meta = json.loads(meta_path.read_text(encoding="utf-8"))
                except Exception:
                    meta = {}
            else:
                no_meta += 1

            title = meta.get("title") or r.get("dataset_title") or ""
            record = {
                "id": vid,
                "link": f"https://www.youtube.com/watch?v={vid}",
                "title": title,
                "description": meta.get("description") or "",
                "channel": meta.get("channel") or "",
                "upload_date": meta.get("upload_date") or "",
                "duration": str(meta.get("duration") or ""),
                "tags": meta.get("tags") or [],
                "thumbnail": meta.get("thumbnail") or "",
                "transcript": segs,
                "source": r.get("source"),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
    print(f"pool.jsonl written to {POOL_JSONL}")
    print(f"  kept: {kept}")
    print(f"  skipped (empty/missing transcript): {skipped_empty_ts}")
    print(f"  missing metadata (transcript-only): {no_meta}")


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("inventory")
    t = sub.add_parser("transcribe")
    t.add_argument("--workers", type=int, default=4)
    m = sub.add_parser("metadata")
    m.add_argument("--workers", type=int, default=8)
    sub.add_parser("assemble")
    args = ap.parse_args()
    {
        "inventory": stage_inventory,
        "transcribe": stage_transcribe,
        "metadata": stage_metadata,
        "assemble": stage_assemble,
    }[args.stage](args)


if __name__ == "__main__":
    main()
