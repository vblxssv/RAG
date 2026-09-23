from pathlib import Path
import fire
from src.indexing import Indexer


class CLI:
    """Command-line interface for the RAG evaluation system."""

    @classmethod
    def run(cls) -> None:
        """Entrypoint for the CLI application."""
        fire.Fire(cls)

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingest data/raw/ and build the index under data/processed/."""
        indexer = Indexer("data/raw", max_chunk_size)
        indexer.run()

    def search(self, query: str, k: int = 5) -> None:
        """Return the top-k sources for a single query."""
        pass

    def search_dataset(
        self,
        dataset_path: str | Path,
        k: int = 5,
        save_directory: str | Path = "data/results",
    ) -> None:
        """Run search over a whole dataset and
        write a StudentSearchResults JSON file."""
        pass

    def answer(self, query: str, k: int = 5) -> None:
        """Answer a single query using the retrieved context."""
        pass

    def answer_dataset(
        self,
        student_search_results_path: str | Path,
        save_directory: str | Path = "data/results",
    ) -> None:
        """Generate answers for a dataset,
        producing a StudentSearchResultsAndAnswer JSON file."""
        pass

    def evaluate(
        self,
        student_search_results_path: str | Path,
        dataset_path: str | Path,
    ) -> None:
        """Report your own recall@k against a ground-truth dataset."""
        pass
