import json
from pathlib import Path
from typing import Any


class BaseStorage:
    """Base storage class encapsulating directory paths and JSON I/O."""

    def __init__(self, base_dir: str | Path) -> None:
        self._dir = Path(base_dir)

    def _ensure_dir(self, directory: Path | None = None) -> Path:
        """Ensures that the directory exists on disk."""
        target = directory or self._dir
        target.mkdir(parents=True, exist_ok=True)
        return target

    def _read_json(self, path: Path) -> Any:
        """Safely reads and deserializes a JSON file with UTF-8 encoding."""
        return json.loads(path.read_text(encoding="utf-8"))

    def _write_json(self, path: Path, data: Any, indent: int = 2) -> None:
        """Safely serializes data to a JSON file ensuring parent exists."""
        self._ensure_dir(path.parent)
        path.write_text(json.dumps(data, indent=indent), encoding="utf-8")
