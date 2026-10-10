from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Self


class FileType(str, Enum):
    """Supported file type categories for chunking strategies."""

    PYTHON = "python"  # .py
    TEXT = "text"     # .txt .rst .md


@dataclass(frozen=True)
class Document:
    path: str
    content: str
    type: FileType

    @cached_property
    def line_starts(self) -> list[int]:
        """Calculate once line shifts."""
        starts = [0]
        for line in self.content.splitlines(keepends=True):
            starts.append(starts[-1] + len(line))
        return starts

    def __len__(self) -> int:
        """Returns the total character count of the document content."""
        return len(self.content)

    @property
    def is_empty(self) -> bool:
        """Returns True if the document has only whitespace or is empty."""
        return not self.content.strip()

    def slice(self, start: int, end: int) -> str:
        """Returns the text slice between start and end character indices."""
        return self.content[start:end]

    def sub_document(self, start: int, end: int) -> Self:
        """Creates a child Document for a sub-range of characters."""
        return self.__class__(
            path=self.path,
            content=self.slice(start, end),
            type=self.type,
        )

    def __str__(self) -> str:
        """Returns a human-readable representation of the document."""
        res = f"Path: {self.path}\nType: {self.type.value}\nContent:\n"
        if len(self.content) > 200:
            res += self.content[:200] + "..."
        else:
            res += self.content
        return res
