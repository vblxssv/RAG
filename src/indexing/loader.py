from enum import Enum
from dataclasses import dataclass
from pathlib import Path
import os
from collections.abc import Iterator


class FileType(str, Enum):
    PYTHON = "python"  # .py
    TEXT = "text"     # .txt .rst .md


@dataclass(frozen=True)
class Document:
    path: str
    content: str
    type: FileType

    def __str__(self) -> str:
        """Returns a human-readable representation of the document."""
        res = f"Path: {self.path}\nType: {self.type.value}\nContent:\n"
        if len(self.content) > 200:
            res += self.content[:200] + "..."
        else:
            res += self.content
        return res


class DocumentLoader:
    def __init__(self, root_dir: str | Path):
        self._root_dir = Path(root_dir)

    def get_files(self) -> list[Path]:
        """Finds all Python and Markdown files in a single fast pass."""
        matched_files: list[Path] = []

        for root, dirs, files in os.walk(self._root_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for file in files:
                if file.endswith((".py", ".md", ".txt", ".rst")):
                    matched_files.append(Path(root, file))
        return matched_files

    def load(self) -> Iterator[Document]:
        """Lazily reads files and yields Document instances one by one."""
        for path in self.get_files():
            doc_type = (
                FileType.PYTHON if path.suffix == ".py" else FileType.TEXT
            )
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            yield Document(
                path=path.as_posix(),
                content=content,
                type=doc_type,
            )
