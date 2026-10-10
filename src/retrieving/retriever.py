"""Retriever module providing lexical search over indexed codebase chunks."""

from typing import List
from pathlib import Path
from tqdm import tqdm
from src.storage import IndexStorage, RetrievingStorage
from src.tokenizer import CodeTokenizer
from src.models import (
    MinimalSource,
    StudentSearchResults,
    MinimalSearchResults,
)


class Retriever:
    """Retrieves top-k relevant source code locations using BM25 index."""

    def __init__(
        self,
        storage: IndexStorage | None = None,
        tokenizer: CodeTokenizer | None = None,
    ) -> None:
        """Initialize retriever by loading sources and index from storage."""
        self._storage = storage or IndexStorage()
        self._tokenizer = tokenizer or CodeTokenizer()
        self._sources, self._index = self._storage.load()

    def search(self, query: str, k: int = 5) -> List[MinimalSource]:
        """Search for top-k source snippets matching the query string."""
        tokenized_query = self._tokenizer.tokenize(query)
        docs = self._index.search(tokenized_query, k)
        return [self._sources[id] for id in docs]

    def search_dataset(
        self,
        dataset_path: str | Path,
        save_directory: str | Path = "data/output/search_results",
        k: int = 5,
    ) -> Path:
        """Execute retrieval across a dataset and save results to disk."""
        retr_stor = RetrievingStorage(dataset_path, save_directory)
        dataset = retr_stor.load_dataset()

        search_results: List[MinimalSearchResults] = []
        for q in tqdm(dataset.rag_questions, desc="Searching questions"):
            sources = self.search(q.question, k)
            search_res = MinimalSearchResults(
                question_id=q.question_id,
                question=q.question,
                retrieved_sources=sources,
            )
            search_results.append(search_res)
        res = StudentSearchResults(k=k, search_results=search_results)
        return retr_stor.save_search_results(res)
