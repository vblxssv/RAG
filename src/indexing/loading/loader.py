from collections.abc import Iterator
import os
from pathlib import Path

from src.indexing.loading.models import Document, FileType


class DocumentLoader:
    """Recursively loads files from a directory as Document instances."""

    def __init__(self, root_dir: str | Path) -> None:
        """Initialize loader with root directory path."""
        self._root_dir = Path(root_dir)
        self._files: list[Path] | None = None

    def get_files(self) -> list[Path]:
        """Finds all Python and Markdown files in a single fast pass."""
        if self._files is not None:
            return self._files
        matched_files: list[Path] = []

        for root, dirs, files in os.walk(self._root_dir):
            dirs[:] = [d for d in dirs if not d.startswith(".")]

            for file in files:
                if file.endswith((".py", ".md", ".txt", ".rst")):
                    matched_files.append(Path(root, file))
        self._files = matched_files
        return self._files

    def __len__(self) -> int:
        """Returns the total number of matched files to load."""
        return len(self.get_files())

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
