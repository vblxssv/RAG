from pathlib import Path
from typing import Any, List

from .chunking import Chunker, MinimalSource
from .loading import DocumentLoader


class Indexer:
    def __init__(self, path: str | Path = "data/raw",
                 max_chunk_size: int = 2000) -> None:
        self._loader = DocumentLoader(path)
        self._chunker = Chunker(max_chunk_size)
        self._sources: List[MinimalSource] = []

    def run(self) -> Any:
        for doc in self._loader.load():
            self._sources.extend(self._chunker.chunk(doc))
        for source in self._sources:
            print('=' * 100)
            print(source)
            print('=' * 100)
