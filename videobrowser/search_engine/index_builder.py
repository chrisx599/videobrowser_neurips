from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Iterable, Optional

from videobrowser.search_engine.base import RETRIEVERS
from videobrowser.search_engine.pool import compute_pool_fingerprint, load_pool
from videobrowser.search_engine.schemas import ENGINE_VERSION, IndexMetadata, PoolRecord

# Ensure retrievers are registered
from videobrowser.search_engine import keyword as _keyword  # noqa: F401
from videobrowser.search_engine import bm25 as _bm25  # noqa: F401
from videobrowser.search_engine import embedding as _embedding  # noqa: F401
from videobrowser.search_engine import hybrid as _hybrid  # noqa: F401


BUILDABLE_METHODS = ("keyword", "bm25", "embedding")


def _load_manifest(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _method_already_current(index_dir: Path, method: str, fingerprint: str) -> bool:
    meta_path = index_dir / method / "meta.json"
    if not meta_path.exists():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return meta.get("fingerprint") == fingerprint and meta.get("engine_version") == ENGINE_VERSION


def build_all(
    pool_path: str | Path,
    index_dir: str | Path,
    methods: Iterable[str],
    fields: list[str],
    embed_model_name: Optional[str] = None,
    embed_batch_size: int = 32,
    bm25_k1: float = 1.5,
    bm25_b: float = 0.75,
    force: bool = False,
    resume: bool = True,
) -> dict:
    """Build requested retriever indexes atomically.

    Writes into a sibling `{index_dir}_tmp/`, then swaps. Skips methods whose
    per-method fingerprint already matches (unless `force=True`).
    """
    pool_path = Path(pool_path)
    index_dir = Path(index_dir)

    records = load_pool(pool_path)
    if not records:
        raise ValueError(f"Pool at {pool_path} yielded zero valid records.")

    fingerprint = compute_pool_fingerprint(records, fields)

    methods_to_build: list[str] = []
    for method in methods:
        if method not in BUILDABLE_METHODS:
            if method == "hybrid":
                print(f"ℹ️ [OfflineSearch] Skipping 'hybrid' build (composes other methods at query time).")
                continue
            raise ValueError(f"Unknown method {method!r}. Known: {BUILDABLE_METHODS}")
        if not force and resume and _method_already_current(index_dir, method, fingerprint):
            print(f"✓ [OfflineSearch] Method {method!r} already current; skipping (use --force to rebuild).")
            continue
        methods_to_build.append(method)

    if not methods_to_build:
        print("ℹ️ [OfflineSearch] Nothing to build.")
        return {"status": "noop", "fingerprint": fingerprint}

    index_dir.mkdir(parents=True, exist_ok=True)

    per_method_meta: dict[str, IndexMetadata] = {}
    for method in methods_to_build:
        cls = RETRIEVERS[method]
        print(f"⏳ [OfflineSearch] Building {method!r} over {len(records)} records...")
        tmp_root = index_dir / f"_tmp_{method}"
        if tmp_root.exists():
            shutil.rmtree(tmp_root)
        tmp_root.mkdir(parents=True, exist_ok=True)

        build_kwargs: dict = {}
        if method == "bm25":
            build_kwargs["k1"] = bm25_k1
            build_kwargs["b"] = bm25_b
        elif method == "embedding":
            if not embed_model_name:
                raise ValueError("embedding build requires --embed-model (or config)")
            build_kwargs["model_name"] = embed_model_name
            build_kwargs["batch_size"] = embed_batch_size

        meta = cls.build(records, tmp_root, fields=fields, **build_kwargs)
        per_method_meta[method] = meta

        final_root = index_dir / method
        if final_root.exists():
            shutil.rmtree(final_root)
        # cls.build wrote under tmp_root / method/; swap directory
        shutil.move(str(tmp_root / method), str(final_root))
        shutil.rmtree(tmp_root, ignore_errors=True)
        print(f"✅ [OfflineSearch] {method!r} built ({meta.doc_count} docs).")

    manifest_path = index_dir / "manifest.json"
    manifest = _load_manifest(manifest_path)
    manifest["engine_version"] = ENGINE_VERSION
    manifest["pool_fingerprint"] = fingerprint
    manifest["fields"] = list(fields)
    manifest["built_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    methods_entry = manifest.setdefault("methods", {})
    for method, meta in per_method_meta.items():
        methods_entry[method] = meta.model_dump()
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {"status": "ok", "fingerprint": fingerprint, "methods": list(per_method_meta.keys())}
