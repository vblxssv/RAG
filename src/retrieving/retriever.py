from src.storage import IndexStorage
from src.tokenizer import CodeTokenizer
from src.models import MinimalSource
from typing import List


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
