from pathlib import Path
from typing import Dict, List, Tuple

from src.bm25 import BM25Indexer
from src.models import MinimalSource
from .base import BaseStorage


class IndexStorage(BaseStorage):
    """Storage gateway for lexical BM25 index and minimal sources."""

    def __init__(self, processed_dir: str | Path = "data/processed") -> None:
        super().__init__(processed_dir)
        self._sources_file = self._dir / "sources.json"
        self._index_file = self._dir / "index.json"
        self._file_cache: Dict[str, str] = {}

    def is_indexed(self) -> bool:
        """Checks whether the index files exist on disk."""
        return self._sources_file.exists() and self._index_file.exists()

    def save(self, sources: List[MinimalSource], index: BM25Indexer) -> None:
        """Persists sources and BM25 index into data/processed/."""
        self._ensure_dir()
        sources_data = [src.model_dump() for src in sources]
        self._write_json(self._sources_file, sources_data)
        self._write_json(self._index_file, index.to_dict())

    def load(self) -> Tuple[List[MinimalSource], BM25Indexer]:
        """Loads sources and the BM25 index from disk."""
        if not self.is_indexed():
            raise FileNotFoundError(
                f"Index not found in {self._dir}. Run 'rag index' first"
            )

        sources_raw = self._read_json(self._sources_file)
        sources = [MinimalSource.model_validate(item) for item in sources_raw]

        index_raw = self._read_json(self._index_file)
        index = BM25Indexer.from_dict(index_raw)

        return sources, index

    def read_snippet(self, source: MinimalSource) -> str:
        """Reads a text slice from the raw file with internal caching."""
        if source.file_path not in self._file_cache:
            self._file_cache[source.file_path] = Path(
                source.file_path
            ).read_text(encoding="utf-8", errors="replace")

        text = self._file_cache[source.file_path]
        return text[source.first_character_index:source.last_character_index]
