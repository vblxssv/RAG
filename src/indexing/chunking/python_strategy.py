import ast
from typing import List

from src.indexing.chunking.base import ChunkStrategy
from src.indexing.chunking.models import MinimalSource, Zone
from src.indexing.chunking.text_strategy import TextChunkStrategy
from src.indexing.loading import Document


class PythonChunkStrategy(ChunkStrategy):
    """Chunking strategy for Python source code using Abstract Syntax Trees."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initializes the Python chunking strategy."""
        super().__init__(max_chunk_size)

    def _get_node_lines(
        self, node: ast.AST, fallback_start: int
    ) -> tuple[int, int]:
        """Returns (start_line, end_line) accounting for decorators."""
        decorators = getattr(node, "decorator_list", None)
        if decorators:
            start = decorators[0].lineno
        else:
            start = getattr(node, "lineno", fallback_start)
        end = getattr(node, "end_lineno", start)
        return start, end

    def _split_class_zone(
        self,
        node: ast.ClassDef,
        class_zone: Zone,
        cursor_line: int,
        document: Document,
    ) -> List[Zone]:
        """Splits a large class into header and method zones."""
        zones: List[Zone] = []
        class_cursor = cursor_line
        class_end_line = getattr(node, "end_lineno", cursor_line)

        for member in getattr(node, "body", []):
            if isinstance(member, (ast.FunctionDef, ast.AsyncFunctionDef)):
                m_start, m_end = self._get_node_lines(member, class_cursor)

                if m_start > class_cursor:
                    h_start = document.line_starts[class_cursor - 1]
                    h_end = document.line_starts[m_start - 1]
                    if h_end > h_start:
                        zones.append(Zone(h_start, h_end))

                m_start_char = document.line_starts[m_start - 1]
                m_end_char = document.line_starts[m_end]
                zones.append(Zone(m_start_char, m_end_char))
                class_cursor = m_end + 1

        if class_cursor <= class_end_line:
            tail_start = document.line_starts[class_cursor - 1]
            tail_end = document.line_starts[class_end_line]
            if tail_end > tail_start:
                zones.append(Zone(tail_start, tail_end))

        return zones

    def _get_zones(self, document: Document) -> List[Zone]:
        tree = ast.parse(document.content)
        cursor_line = 1
        zones: List[Zone] = []

        for node in getattr(tree, "body", []):
            _, end_line = self._get_node_lines(node, cursor_line)
            start_char = document.line_starts[cursor_line - 1]
            end_char = document.line_starts[end_line]

            if isinstance(node, ast.ClassDef):
                class_zone = Zone(start_char, end_char)
                if (
                    class_zone.end_char - class_zone.start_char
                    > self.max_chunk_size
                ):
                    class_zones: List[Zone] = self._split_class_zone(
                        node, class_zone, cursor_line, document
                    )
                    zones.extend(class_zones)
                else:
                    zones.append(class_zone)
            else:
                zones.append(Zone(start_char, end_char))
            cursor_line = end_line + 1

        total_chars = len(document.content)
        last_char = zones[-1].end_char if zones else 0
        if last_char < total_chars:
            zones.append(Zone(last_char, total_chars))

        return zones

    def _slice_large_zone(
        self, document: Document, zone: Zone
    ) -> List[MinimalSource]:
        """Slices an oversized zone using text strategy."""
        sub_text = document.content[zone.start_char:zone.end_char]
        sub_doc = Document(document.path, sub_text, document.type)
        return [
            MinimalSource(
                file_path=document.path,
                first_character_index=(
                    zone.start_char + s.first_character_index
                ),
                last_character_index=(
                    zone.start_char + s.last_character_index
                ),
            )
            for s in TextChunkStrategy(self.max_chunk_size).chunk(sub_doc)
        ]

    def chunk(self, document: Document) -> List[MinimalSource]:
        text = document.content
        if not text.strip():
            return []
        if len(text) <= self.max_chunk_size:
            return [
                MinimalSource(
                    file_path=document.path,
                    first_character_index=0,
                    last_character_index=len(text),
                )
            ]

        try:
            zones = self._get_zones(document)
        except (SyntaxError, ValueError):
            return TextChunkStrategy(self.max_chunk_size).chunk(document)

        sources: List[MinimalSource] = []
        start: int | None = None
        end: int | None = None

        for z in zones:
            if (z.end_char - z.start_char) > self.max_chunk_size:
                if start is not None and end is not None:
                    sources.append(
                        MinimalSource(
                            file_path=document.path,
                            first_character_index=start,
                            last_character_index=end,
                        )
                    )
                    start, end = None, None
                sources.extend(self._slice_large_zone(document, z))
            elif start is None:
                start, end = z.start_char, z.end_char
            elif (z.end_char - start) <= self.max_chunk_size:
                end = z.end_char
            elif start is not None and end is not None:
                sources.append(
                    MinimalSource(
                        file_path=document.path,
                        first_character_index=start,
                        last_character_index=end,
                    )
                )
                start, end = z.start_char, z.end_char

        if start is not None and end is not None:
            sources.append(
                MinimalSource(
                    file_path=document.path,
                    first_character_index=start,
                    last_character_index=end,
                )
            )

        return sources
