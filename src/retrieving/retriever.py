from src.storage import IndexStorage, RetrievingStorage
from src.tokenizer import CodeTokenizer
from src.models import (MinimalSource, StudentSearchResults,
                        MinimalSearchResults)
from typing import List
from pathlib import Path
from tqdm import tqdm


class Retriever:
    def __init__(self, storage: IndexStorage | None = None,
                 tokenizer: CodeTokenizer | None = None) -> None:
        self._storage = storage or IndexStorage()
        self._tokenizer = tokenizer or CodeTokenizer()
        self._sources, self._index = self._storage.load()

    def search(self, query: str, k: int = 5) -> List[MinimalSource]:
        tokenized_query = self._tokenizer.tokenize(query)

        docs = self._index.search(tokenized_query, k)

        return [self._sources[id] for id in docs]

    def search_dataset(
        self,
        dataset_path: str | Path,
        save_directory: str | Path = "data/output/search_results",
        k: int = 5,
    ) -> Path:
        retr_stor = RetrievingStorage(dataset_path,
                                      save_directory)
        dataset = retr_stor.load_dataset()

        search_results: List[MinimalSearchResults] = []
        for q in tqdm(dataset.rag_questions, desc="Searching questions"):
            sources = self.search(q.question, k)
            search_res = MinimalSearchResults(question_id=q.question_id,
                                              question=q.question,
                                              retrieved_sources=sources)
            search_results.append(search_res)
        res = StudentSearchResults(k=k,
                                   search_results=search_results)
        p = retr_stor.save_search_results(res)
        return p
