from src.indexing.chunking.strategies import (
    Chunk,
    ChunkStrategy,
    PythonChunkStrategy,
    TextChunkStrategy,
)
from src.indexing.loader import Document, FileType


class Chunker:
    """Dispatches documents to the appropriate chunking strategy."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initializes Chunker with strategies for each supported FileType.

        Args:
            max_chunk_size: Maximum chunk size in characters.
        """
        self.max_chunk_size = max_chunk_size
        self._strategies: dict[FileType, ChunkStrategy] = {
            FileType.PYTHON: PythonChunkStrategy(max_chunk_size),
            FileType.TEXT: TextChunkStrategy(max_chunk_size),
        }

    def chunk(self, document: Document) -> list[Chunk]:
        """Splits a document into chunks using its corresponding strategy.

        Args:
            document: Document instance to chunk.

        Returns:
            List of generated Chunk objects.
        """
        strategy = self._strategies[document.type]
        return strategy.chunk(document)
