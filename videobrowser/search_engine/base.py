from __future__ import annotations

from pathlib import Path
from typing import ClassVar, Protocol, runtime_checkable

from videobrowser.search_engine.schemas import IndexMetadata, PoolRecord, SearchHit


class SearchMethodNotBuilt(RuntimeError):
    def __init__(self, method: str, index_dir: Path | str | None = None):
        self.method = method
        hint = (
            f"python -m videobrowser.search_engine.build_index --methods {method} --pool <pool.jsonl> --index-dir <index_dir>"
        )
        extra = f" Expected artifacts missing under {index_dir}/{method}/." if index_dir else ""
        super().__init__(
            f"Offline search method {method!r} is not built.{extra} Rebuild with: {hint}"
        )


@runtime_checkable
class Retriever(Protocol):
    name: ClassVar[str]

    def load(self, index_dir: Path) -> None: ...
    def search(self, query: str, k: int) -> list[SearchHit]: ...


class RetrieverBuilder(Protocol):
    @classmethod
    def build(
        cls,
        records: list[PoolRecord],
        index_dir: Path,
        fields: list[str],
        **kwargs,
    ) -> IndexMetadata: ...


RETRIEVERS: dict[str, type] = {}


def register(cls: type):
    RETRIEVERS[cls.name] = cls
    return cls
