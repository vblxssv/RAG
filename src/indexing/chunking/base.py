from abc import ABC, abstractmethod
from typing import List

from src.indexing.chunking.models import MinimalSource
from src.indexing.loading import Document


class ChunkStrategy(ABC):
    """Abstract base class for document chunking strategies."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        self.max_chunk_size = max_chunk_size

    @abstractmethod
    def chunk(self, document: Document) -> List[MinimalSource]:
        """Splits a document into a list of minimal sources."""
        pass
