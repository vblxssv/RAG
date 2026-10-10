from pathlib import Path
from tqdm import tqdm
from .qwen import Qwen
from src.retrieving import Retriever
from src.storage import IndexStorage, RetrievingStorage, AnswerStorage
from src.models import (
    MinimalSource,
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
)


class AnswerGenerator:
    """Orchestrates RAG answer generation for single queries and datasets."""

    def __init__(
        self,
        llm: Qwen | None = None,
        retriever: Retriever | None = None,
        index_storage: IndexStorage | None = None,
    ) -> None:
        """Initialize core generation dependencies."""
        self._llm = llm or Qwen()
        self._retriever = retriever or Retriever()
        self._index_storage = index_storage or IndexStorage()

    def _get_chunks(self, sources: list[MinimalSource]) -> list[str]:
        """Extract text snippets from corpus files for all sources."""
        return [self._index_storage.read_snippet(src) for src in sources]

    def _build_context(self, sources: list[MinimalSource]) -> list[str]:
        """Extract text snippets from corpus files for all sources."""
        return self._get_chunks(sources)

    def answer(self, query: str, k: int = 5) -> str:
        """Answer a single query on the fly."""
        sources = self._retriever.search(query, k)
        chunks = self._get_chunks(sources)
        return self._llm.generate(question=query, chunks=chunks)

    def answer_dataset(
        self,
        student_search_results_path: str | Path,
        save_directory: str | Path,
    ) -> Path:
        """Generate answers for a whole dataset using pre-retrieved sources."""
        search_path = Path(student_search_results_path)

        retr_storage = RetrievingStorage(search_path, search_path.parent)
        search_results = retr_storage.load_search_results(search_path)
        ans_storage = AnswerStorage(save_directory)

        answers: list[MinimalAnswer] = []
        for item in tqdm(search_results.search_results,
                         desc="Generating answers"):
            chunks = self._get_chunks(item.retrieved_sources)
            ans_text = self._llm.generate(
                question=item.question,
                chunks=chunks,
            )

            answers.append(
                MinimalAnswer(
                    question_id=item.question_id,
                    question=item.question,
                    retrieved_sources=item.retrieved_sources,
                    answer=ans_text,
                )
            )

        final_result = StudentSearchResultsAndAnswer(
            search_results=answers,
            k=search_results.k,
        )
        return ans_storage.save(final_result, search_path.name)
