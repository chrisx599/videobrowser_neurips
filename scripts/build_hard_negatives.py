"""Build hard-negative candidates for the offline 450-question slice.

Subcommands (run individually, idempotent, --resume safe):
    search                – Stage 1: YouTube search via yt-dlp
    prefilter             – Stage 2: pure local filter
    layer1                – Stage 3: metadata+transcript LLM verify
    download_transcribe   – Stage 4a: download + whisper
    layer2                – Stage 4b: sparse-frame VLM verify
    select                – Stage 5: rank + take top-N
    integrate             – Stage 6: emit pool_with_hard_negatives + sidecar
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from videobrowser.hard_negatives import storage  # noqa: E402

OUT_ROOT = ROOT / "data/offline_search/hard_negatives"


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search")
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--top-k", type=int, default=20)

    pf = sub.add_parser("prefilter")
    pf.add_argument("--limit", type=int, default=0)

    l1 = sub.add_parser("layer1")
    l1.add_argument("--limit", type=int, default=0)
    l1.add_argument("--workers", type=int, default=4)

    dl = sub.add_parser("download_transcribe")
    dl.add_argument("--limit", type=int, default=0)
    dl.add_argument("--workers", type=int, default=4)
    dl.add_argument("--whisper-endpoints", default="http://localhost:8028,http://localhost:8038",
                    help="Comma-separated whisper endpoint base URLs.")
    dl.add_argument("--per-question-cap", type=int, default=0,
                    help="Keep only the top-K layer-1 survivors per question "
                         "(ranked by overlap_score, layer-1 relevance, yt_rank). "
                         "0 = no cap.")

    l2 = sub.add_parser("layer2")
    l2.add_argument("--limit", type=int, default=0)
    l2.add_argument("--workers", type=int, default=2)
    l2.add_argument("--n-frames", type=int, default=16)

    sl = sub.add_parser("select")
    sl.add_argument("--n", type=int, default=3)

    sub.add_parser("integrate")

    args = p.parse_args()
    if args.cmd == "search":
        from scripts._hardneg_search import run_search
        run_search(out_root=OUT_ROOT, limit=args.limit,
                   workers=args.workers, top_k=args.top_k)
    elif args.cmd == "prefilter":
        from scripts._hardneg_prefilter import run_prefilter
        run_prefilter(out_root=OUT_ROOT, limit=args.limit)
    elif args.cmd == "layer1":
        from scripts._hardneg_layer1 import run_layer1
        run_layer1(out_root=OUT_ROOT, limit=args.limit, workers=args.workers)
    elif args.cmd == "download_transcribe":
        from scripts._hardneg_download import run_download_transcribe
        run_download_transcribe(
            out_root=OUT_ROOT,
            limit=args.limit, workers=args.workers,
            whisper_endpoints=[e.strip() for e in args.whisper_endpoints.split(",") if e.strip()],
            per_question_cap=args.per_question_cap,
        )
    elif args.cmd == "layer2":
        from scripts._hardneg_layer2 import run_layer2
        run_layer2(out_root=OUT_ROOT, limit=args.limit,
                   workers=args.workers, n_frames=args.n_frames)
    elif args.cmd == "select":
        from scripts._hardneg_select import run_select
        run_select(out_root=OUT_ROOT, n=args.n)
    elif args.cmd == "integrate":
        from scripts._hardneg_select import run_integrate
        run_integrate(out_root=OUT_ROOT)
    else:
        raise SystemExit(f"unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
