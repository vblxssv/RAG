from src.indexing.chunking.base import ChunkStrategy
from src.indexing.chunking.models import Chunk
from src.indexing.chunking.python_strategy import PythonChunkStrategy
from src.indexing.chunking.text_strategy import TextChunkStrategy
from src.indexing.loading import Document, FileType


class Chunker:
    """Dispatches documents to the appropriate chunking strategy."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """
            Initializes Chunker with strategies for each supported FileType.
        """
        self.max_chunk_size = max_chunk_size
        self._strategies: dict[FileType, ChunkStrategy] = {
            FileType.PYTHON: PythonChunkStrategy(max_chunk_size),
            FileType.TEXT: TextChunkStrategy(max_chunk_size),
        }

    def chunk(self, document: Document) -> list[Chunk]:
        """
            Splits a document into chunks using its corresponding strategy.
        """
        strategy = self._strategies[document.type]
        return strategy.chunk(document)
