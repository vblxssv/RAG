"""Command-line interface module defining all user-facing commands."""

from pathlib import Path
from typing import Annotated
from pydantic import Field, validate_call
from src.indexing import Indexer
from src.retrieving import Retriever
from src.recall import Recall
from src.generating import AnswerGenerator

NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(gt=0)]


class CLI:
    """Command-line interface for the RAG evaluation system."""

    @validate_call
    def index(
        self,
        max_chunk_size: PositiveInt = 2000,
        raw_dir: str | Path = "data/raw",
    ) -> None:
        """Ingest corpus files and build the lexical search index."""
        Indexer(raw_dir, max_chunk_size).run()
        print("Ingestion complete! Indices saved under data/processed/")

    @validate_call
    def search(self, query: str, k: NonNegativeInt = 5) -> None:
        """Return the top-k sources for a single query."""
        retriever = Retriever()
        results = retriever.search(query, k)
        for res in results:
            print(res)

    @validate_call
    def search_dataset(
        self,
        dataset_path: str | Path,
        k: NonNegativeInt = 5,
        save_directory: str | Path = "data/output/search_results",
    ) -> None:
        """Run search over a whole dataset and save search results."""
        dataset_path = Path(dataset_path)
        save_dir = Path(save_directory)
        if save_dir == Path("data/output/search_results"):
            save_dir = save_dir / dataset_path.parent.name

        retriever = Retriever()
        p = retriever.search_dataset(dataset_path, save_dir, k)
        print(f"Saved student_search_results to {p.as_posix()}")

    @validate_call
    def answer(self, query: str, k: NonNegativeInt = 5) -> None:
        """Answer a single query using the retrieved context."""
        print(AnswerGenerator().answer(query, k))

    @validate_call
    def answer_dataset(
        self,
        student_search_results_path: str | Path,
        save_directory: str | Path = "data/output/search_results_and_answer",
    ) -> None:
        """Generate answers for a dataset from pre-retrieved sources."""
        results_path = Path(student_search_results_path)
        save_dir = Path(save_directory)
        if save_dir == Path("data/output/search_results_and_answer"):
            save_dir = save_dir / results_path.parent.name

        gen = AnswerGenerator()
        p = gen.answer_dataset(results_path, save_dir)
        print(f"Saved student_search_results_and_answer to {p.as_posix()}")

    @validate_call
    def evaluate(
        self,
        student_search_results_path: str | Path,
        dataset_path: str | Path,
    ) -> None:
        """Report recall@k metrics against a ground-truth dataset."""
        recall = Recall(student_search_results_path, dataset_path)
        print("Evaluation Results")
        print("=" * 40)
        for k in (1, 3, 5, 10):
            score = recall.calculate_at_k(k)
            print(f"Recall@{k}: {score:.3f} ({score * 100:.1f}%)")
