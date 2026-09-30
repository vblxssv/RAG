from pathlib import Path
from typing import Any, List

from .chunking import Chunk, Chunker
from .loading import DocumentLoader


class Indexer:
    def __init__(self, path: str | Path = "data/raw",
                 max_chunk_size: int = 2000) -> None:
        self._loader = DocumentLoader(path)
        self._chunker = Chunker(max_chunk_size)
        self._chunks: List[Chunk] = []

    def run(self) -> Any:
        for doc in self._loader.load():
            self._chunks.extend(self._chunker.chunk(doc))
        for chunk in self._chunks:
            print('=' * 100)
            print(chunk)
            print('=' * 100)
