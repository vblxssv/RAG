from pathlib import Path
import fire
from src.indexing import Indexer
from src.retrieving import Retriever
from src.recall import Recall


class CLI:
    """Command-line interface for the RAG evaluation system."""

    @classmethod
    def run(cls) -> None:
        """Entrypoint for the CLI application."""
        fire.Fire(cls)

    def index(self, max_chunk_size: int = 2000) -> None:
        """Ingest data/raw/ and build the index under data/processed/."""
        Indexer("data/raw", max_chunk_size).run()
        print("Ingestion complete! Indices saved under data/processed/")

    def search(self, query: str, k: int = 5) -> None:
        """Return the top-k sources for a single query."""
        retriever = Retriever()
        results = retriever.search(query, k)
        for res in results:
            print(res)

    def search_dataset(
        self,
        dataset_path: str | Path,
        k: int = 5,
        save_directory: str | Path = "data/output/search_results",
    ) -> None:
        """Run search over a whole dataset and
        write a StudentSearchResults JSON file."""
        dataset_path = Path(dataset_path)
        save_dir = Path(save_directory)
        if save_dir == Path("data/output/search_results"):
            save_dir = save_dir / dataset_path.parent.name

        retriever = Retriever()
        p = retriever.search_dataset(dataset_path, save_dir, k)
        print(f"Saved student_search_results to {p.as_posix()}")

    def answer(self, query: str, k: int = 5) -> None:
        """Answer a single query using the retrieved context."""
        pass

    def answer_dataset(
        self,
        student_search_results_path: str | Path,
        save_directory: str | Path = "data/output/search_results_and_answer",
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
        recall = Recall(student_search_results_path, dataset_path)
        print("Evaluation Results")
        print("=" * 40)
        for k in (1, 3, 5, 10):
            score = recall.calculate_at_k(k)
            print(f"Recall@{k}: {score:.3f} ({score * 100:.1f}%)")
