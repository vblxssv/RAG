from .base import BaseStorage
from pathlib import Path
from src.models import StudentSearchResultsAndAnswer


class AnswerStorage(BaseStorage):
    """Storage responsible for persisting and reading generated answers."""

    def __init__(self, results_dir: str | Path) -> None:
        super().__init__(results_dir)
        self._results_dir = Path(results_dir)

    def save(self,
             answers: StudentSearchResultsAndAnswer, filename: str) -> Path:
        self._ensure_dir(self._results_dir)
        out_path = self._results_dir / filename
        out_path.write_text(answers.model_dump_json(indent=2),
                            encoding="utf-8")
        return out_path

    def load(self, path: str | Path) -> StudentSearchResultsAndAnswer:
        raw_text = Path(path).read_text(encoding="utf-8")
        return StudentSearchResultsAndAnswer.model_validate_json(raw_text)
