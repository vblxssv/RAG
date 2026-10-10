from typing import List

from src.indexing.chunking.base import ChunkStrategy
from src.indexing.chunking.models import MinimalSource
from src.indexing.loading import Document


class TextChunkStrategy(ChunkStrategy):
    """Chunking strategy for plain text and markdown documents."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        super().__init__(max_chunk_size)

    def _find_end(self, text: str, start: int, limit: int) -> int:
        """Finds the best cutoff point between start and limit."""
        if limit == len(text):
            return limit
        best_pos = -1
        for sep in (". ", ".\n", "! ", "? ", "\n\n", "\n"):
            pos = text.rfind(sep, start, limit)
            if pos != -1:
                cutoff = pos + (1 if sep[0] in ".!?" else len(sep))
                if cutoff > best_pos:
                    best_pos = cutoff
        if best_pos > start:
            return best_pos
        space_pos = text.rfind(" ", start, limit)
        if space_pos > start:
            return space_pos
        return limit

    def chunk(self, document: Document) -> List[MinimalSource]:
        """Split plain text or markdown document into chunks."""
        if document.is_empty:
            return []
        sources: List[MinimalSource] = []
        text: str = document.content
        size: int = len(document)

        cursor = 0
        while cursor < size:
            while cursor < size and text[cursor].isspace():
                cursor += 1
            if cursor >= size:
                break
            sentence_start = cursor
            limit = min(sentence_start + self.max_chunk_size, size)
            chunk_end = self._find_end(text, sentence_start, limit)
            sources.append(
                MinimalSource(
                    file_path=document.path,
                    first_character_index=sentence_start,
                    last_character_index=chunk_end,
                )
            )
            cursor = chunk_end
        return sources
