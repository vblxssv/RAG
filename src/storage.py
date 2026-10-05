import json
from pathlib import Path
from typing import Dict, List, Tuple

from src.bm25 import BM25Indexer
from src.models import MinimalSource


class IndexStorage:
    def __init__(self, processed_dir: str | Path = "data/processed") -> None:
        self._dir = Path(processed_dir)
        self._sources_file = self._dir / "sources.json"
        self._index_file = self._dir / "index.json"
        self._file_cache: Dict[str, str] = {}

    def is_indexed(self) -> bool:
        return self._sources_file.exists() and self._index_file.exists()

    def save(self, sources: List[MinimalSource], index: BM25Indexer) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)

        sources_data = [src.model_dump() for src in sources]
        self._sources_file.write_text(
            json.dumps(sources_data), encoding="utf-8"
        )
        self._index_file.write_text(
            json.dumps(index.to_dict()), encoding="utf-8"
        )

    def load(self) -> Tuple[List[MinimalSource], BM25Indexer]:
        if not self.is_indexed():
            raise FileNotFoundError(
                f"Индекс не найден в {self._dir}. Запустите 'rag index'"
            )

        sources_raw = json.loads(
            self._sources_file.read_text(encoding="utf-8")
        )
        sources = [MinimalSource.model_validate(item) for item in sources_raw]

        index_raw = json.loads(self._index_file.read_text(encoding="utf-8"))
        index = BM25Indexer.from_dict(index_raw)

        return sources, index

    def read_snippet(self, source: MinimalSource) -> str:
        if source.file_path not in self._file_cache:
            self._file_cache[source.file_path] = Path(
                source.file_path
            ).read_text(encoding="utf-8", errors="replace")

        text = self._file_cache[source.file_path]
        return text[source.first_character_index:source.last_character_index]
