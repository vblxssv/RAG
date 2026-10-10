"""Corpus indexing orchestrator."""

from pathlib import Path
from typing import Any, List

from tqdm import tqdm

from src.bm25 import BM25Indexer
from src.models import MinimalSource
from src.storage import IndexStorage
from src.tokenizer import CodeTokenizer
from .chunking import Chunker
from .loading import DocumentLoader


class Indexer:
    """Chunks corpus files and builds a searchable BM25 index."""

    def __init__(
        self,
        path: str | Path = "data/raw",
        max_chunk_size: int = 2000,
    ) -> None:
        """Initialize indexing components and storage."""
        self._loader = DocumentLoader(path)
        self._chunker = Chunker(max_chunk_size)
        self._tokenizer = CodeTokenizer()
        self._storage = IndexStorage()

    def run(self) -> Any:
        """Build BM25 index over documents and persist to disk."""
        sources: List[MinimalSource] = []
        tokens: List[List[str]] = []
        for doc in tqdm(
            self._loader.load(),
            total=len(self._loader),
            desc="Indexing documents",
            ascii=" ╸━",
            colour="cyan",
            bar_format="{desc} [{bar:40}] {n_fmt}/{total_fmt} [{elapsed}]",
        ):
            srcs = self._chunker.chunk(doc)
            for src in srcs:
                chunk_text = doc.slice(
                    src.first_character_index,
                    src.last_character_index,
                )
                cur_tokens = self._tokenizer.tokenize(chunk_text)
                tokens.append(cur_tokens)
                sources.append(src)
        index = BM25Indexer.build(tokens)
        self._storage.save(sources, index)
