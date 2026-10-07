from .base import BaseStorage
from pathlib import Path
from src.models import RagDataset, StudentSearchResults


class RetrievingStorage(BaseStorage):
    def __init__(self, input_path: str | Path,
                 results_dir: str | Path) -> None:
        super().__init__(input_path)
        self._input_path = Path(input_path)
        self._results_dir = Path(results_dir)

    def load_dataset(self) -> RagDataset:
        if not self._input_path.exists():
            raise FileNotFoundError(f"File: {self._input_path} does not exist")
        raw_text = self._input_path.read_text(encoding="utf-8")
        dataset = RagDataset.model_validate_json(raw_text)
        return dataset

    def save_search_results(self,
                            search_results: StudentSearchResults) -> Path:
        self._ensure_dir(self._results_dir)
        result_path = self._results_dir / self._input_path.name
        json_text = search_results.model_dump_json(indent=2)
        result_path.write_text(json_text, encoding="utf-8")
        return result_path

    def load_search_results(self) -> StudentSearchResults:
        res_path = self._results_dir / self._input_path.name
        if not res_path.exists():
            raise FileNotFoundError(f"File: {res_path} does not exist")
        raw_text = res_path.read_text(encoding="utf-8")
        search_results = StudentSearchResults.model_validate_json(raw_text)
        return search_results
