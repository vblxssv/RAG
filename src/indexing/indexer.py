from .loader import DocumentLoader
from .chunking import Chunker
from pathlib import Path
from typing import Any


class Indexer:
    def __init__(self, path: str | Path = "data/raw",
                 max_chunk_size: int = 2000) -> None:
        self._loader = DocumentLoader(path)
        self._chunker = Chunker(max_chunk_size=max_chunk_size)

    def run(self) -> Any:
        chunks = []
        for doc in self._loader.load():
            chunks.extend(self._chunker.chunk(doc))

        for ch in chunks:
            print(ch)
            print("=" * 80)
