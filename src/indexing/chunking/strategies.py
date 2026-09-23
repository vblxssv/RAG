import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from src.indexing.loader import Document


@dataclass(frozen=True)
class Chunk:
    """A segment of a document with exact character coordinates."""
    file_path: str
    content: str
    first_character_index: int
    last_character_index: int

    def __str__(self) -> str:
        res = ''
        res += (f"chunk: {self.file_path}"
                f"[{self.first_character_index}:"
                f"{self.last_character_index}]\n")
        res += self.content
        return res


class ChunkStrategy(ABC):
    def __init__(self, max_chunk_size: int = 2000) -> None:
        self.max_chunk_size = max_chunk_size

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        pass


class TextChunkStrategy(ChunkStrategy):
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

    def chunk(self, document: Document) -> List[Chunk]:
        chunks: List[Chunk] = []
        text: str = document.content
        size: int = len(text)

        cursor = 0
        while cursor < size:
            while cursor < size and text[cursor].isspace():
                cursor += 1
            if cursor >= size:
                break
            sentence_start = cursor
            limit = min(sentence_start + self.max_chunk_size, size)
            chunk_end = self._find_end(text, sentence_start, limit)
            chunks.append(
                Chunk(document.path,
                      text[sentence_start:chunk_end],
                      sentence_start, chunk_end)
                )
            cursor = chunk_end
        return chunks


class PythonChunkStrategy(ChunkStrategy):
    """Chunking strategy for Python source code using Abstract Syntax Trees."""

    def __init__(self, max_chunk_size: int = 2000) -> None:
        """Initializes the Python chunking strategy.

        Args:
            max_chunk_size: Maximum chunk size in characters.
        """
        super().__init__(max_chunk_size)

    def _get_node_span(
        self,
        node: ast.AST,
        line_offsets: list[int],
        total_len: int,
    ) -> tuple[int, int]:
        """Calculates start and end character offsets for an AST node."""
        start_line = getattr(node, "lineno", 1)
        if hasattr(node, "decorator_list") and node.decorator_list:
            start_line = min(start_line, node.decorator_list[0].lineno)
        end_line = getattr(node, "end_lineno", start_line)

        start_char = line_offsets[start_line - 1]
        if end_line < len(line_offsets):
            end_char = line_offsets[end_line]
        else:
            end_char = total_len
        return start_char, end_char

    def _split_by_lines(
        self,
        file_path: str,
        text: str,
        start_char: int,
        end_char: int,
    ) -> list[Chunk]:
        """Splits an oversized block of code line by line."""
        sub = text[start_char:end_char]
        lines = sub.splitlines(keepends=True)
        chunks: list[Chunk] = []
        curr_start = start_char
        curr_len = 0

        for line in lines:
            line_len = len(line)
            if line_len > self.max_chunk_size:
                if curr_len > 0:
                    chunks.append(
                        Chunk(
                            file_path=file_path,
                            content=text[curr_start:curr_start + curr_len],
                            first_character_index=curr_start,
                            last_character_index=curr_start + curr_len,
                        )
                    )
                    curr_start += curr_len
                    curr_len = 0
                for i in range(0, line_len, self.max_chunk_size):
                    chunk_slice = line[i:i + self.max_chunk_size]
                    s_idx = curr_start + i
                    e_idx = s_idx + len(chunk_slice)
                    chunks.append(
                        Chunk(
                            file_path=file_path,
                            content=chunk_slice,
                            first_character_index=s_idx,
                            last_character_index=e_idx,
                        )
                    )
                curr_start += line_len
            elif curr_len + line_len > self.max_chunk_size and curr_len > 0:
                chunks.append(
                    Chunk(
                        file_path=file_path,
                        content=text[curr_start:curr_start + curr_len],
                        first_character_index=curr_start,
                        last_character_index=curr_start + curr_len,
                    )
                )
                curr_start += curr_len
                curr_len = line_len
            else:
                curr_len += line_len

        if curr_len > 0:
            chunks.append(
                Chunk(
                    file_path=file_path,
                    content=text[curr_start:curr_start + curr_len],
                    first_character_index=curr_start,
                    last_character_index=curr_start + curr_len,
                )
            )
        return chunks

    def chunk(self, document: Document) -> list[Chunk]:
        """Chunks a Python document by AST nodes, preserving code structure."""
        text = document.content
        if not text.strip():
            return []

        try:
            tree = ast.parse(text)
        except SyntaxError:
            fallback = TextChunkStrategy(self.max_chunk_size)
            return fallback.chunk(document)

        if not tree.body:
            fallback = TextChunkStrategy(self.max_chunk_size)
            return fallback.chunk(document)

        line_offsets = [0]
        for line in text.splitlines(keepends=True):
            line_offsets.append(line_offsets[-1] + len(line))

        total_len = len(text)
        raw_spans: list[tuple[int, int]] = []

        for node in tree.body:
            sc, ec = self._get_node_span(node, line_offsets, total_len)
            if isinstance(node, ast.ClassDef) and (
                ec - sc > self.max_chunk_size
            ):
                first_method_start = ec
                method_spans: list[tuple[int, int]] = []
                for child in node.body:
                    csc, cec = self._get_node_span(
                        child, line_offsets, total_len
                    )
                    if isinstance(
                        child, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ):
                        if csc < first_method_start:
                            first_method_start = csc
                        method_spans.append((csc, cec))
                    else:
                        method_spans.append((csc, cec))

                if first_method_start > sc:
                    raw_spans.append((sc, first_method_start))
                raw_spans.extend(method_spans)
            else:
                raw_spans.append((sc, ec))

        chunks: list[Chunk] = []
        curr_start = -1
        curr_end = -1

        for sc, ec in raw_spans:
            if ec - sc > self.max_chunk_size:
                if curr_start != -1:
                    chunks.append(
                        Chunk(
                            file_path=document.path,
                            content=text[curr_start:curr_end],
                            first_character_index=curr_start,
                            last_character_index=curr_end,
                        )
                    )
                    curr_start = -1
                    curr_end = -1
                chunks.extend(
                    self._split_by_lines(document.path, text, sc, ec)
                )
            else:
                if curr_start == -1:
                    curr_start = sc
                    curr_end = ec
                elif ec - curr_start <= self.max_chunk_size:
                    curr_end = ec
                else:
                    chunks.append(
                        Chunk(
                            file_path=document.path,
                            content=text[curr_start:curr_end],
                            first_character_index=curr_start,
                            last_character_index=curr_end,
                        )
                    )
                    curr_start = sc
                    curr_end = ec

        if curr_start != -1:
            chunks.append(
                Chunk(
                    file_path=document.path,
                    content=text[curr_start:curr_end],
                    first_character_index=curr_start,
                    last_character_index=curr_end,
                )
            )

        return chunks
