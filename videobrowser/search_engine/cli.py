from __future__ import annotations

import argparse
import sys
from typing import Optional

from videobrowser.search_engine.index_builder import build_all


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="videobrowser.search_engine.build_index",
        description="Build pre-computed indexes for the offline video search engine.",
    )
    parser.add_argument("--pool", required=True, help="Path to pool JSONL file.")
    parser.add_argument("--index-dir", required=True, help="Directory to write indexes.")
    parser.add_argument(
        "--methods",
        default="bm25",
        help="Comma-separated list of methods to build (keyword,bm25,embedding).",
    )
    parser.add_argument(
        "--fields",
        default="title,description,channel,tags,transcript",
        help="Comma-separated list of record fields to include in indexed text.",
    )
    parser.add_argument(
        "--embed-model",
        default=None,
        help="SentenceTransformer model path/name for the embedding method.",
    )
    parser.add_argument("--embed-batch-size", type=int, default=32)
    parser.add_argument("--bm25-k1", type=float, default=1.5)
    parser.add_argument("--bm25-b", type=float, default=0.75)
    parser.add_argument("--force", action="store_true", help="Rebuild even if fingerprint matches.")
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        default=True,
        help="Disable resume (default: resume).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    fields = [f.strip() for f in args.fields.split(",") if f.strip()]

    try:
        result = build_all(
            pool_path=args.pool,
            index_dir=args.index_dir,
            methods=methods,
            fields=fields,
            embed_model_name=args.embed_model,
            embed_batch_size=args.embed_batch_size,
            bm25_k1=args.bm25_k1,
            bm25_b=args.bm25_b,
            force=args.force,
            resume=args.resume,
        )
    except FileNotFoundError as exc:
        print(f"❌ Pool missing: {exc}")
        return 2
    except ValueError as exc:
        print(f"❌ {exc}")
        return 2
    except Exception as exc:
        print(f"❌ Index build failed: {exc}")
        return 3

    print(f"Done: {result}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
