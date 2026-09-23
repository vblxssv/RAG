from .loader import DocumentLoader
from pathlib import Path
from typing import Any


class Indexer:
    def __init__(self, path: str | Path = "data/raw",
                 max_chunk_size: int = 2000) -> None:
        self._loader = DocumentLoader(path)

    def run(self) -> Any:
        for doc in self._loader.load():
            print(doc)
            print("=" * 40)
