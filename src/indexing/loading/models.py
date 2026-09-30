from dataclasses import dataclass
from enum import Enum
from functools import cached_property


class FileType(str, Enum):
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

    def __str__(self) -> str:
        """Returns a human-readable representation of the document."""
        res = f"Path: {self.path}\nType: {self.type.value}\nContent:\n"
        if len(self.content) > 200:
            res += self.content[:200] + "..."
        else:
            res += self.content
        return res
