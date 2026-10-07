from .base import BaseStorage
from pathlib import Path


class RetrievingStorage(BaseStorage):
    def __init__(self, input_dir: str | Path, results_dir: str | Path):
        super().__init__(input)
        self._input_dir = input_dir
        self._results_dir = results_dir
