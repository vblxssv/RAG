from pathlib import Path
from tqdm import tqdm
from .llm import QwenLLM
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
        llm: QwenLLM | None = None,
        retriever: Retriever | None = None,
        index_storage: IndexStorage | None = None,
    ) -> None:
        """Initialize core generation dependencies."""
        self._llm = llm or QwenLLM()
        self._retriever = retriever or Retriever()
        self._index_storage = index_storage or IndexStorage()

    def _build_context(self, sources: list[MinimalSource]) -> str:
        """Extract text snippets from corpus files for all sources."""
        snippets = [self._index_storage.read_snippet(src) for src in sources]
        return "\n\n---\n\n".join(snippets)

    def answer(self, query: str, k: int = 5) -> str:
        """Answer a single query on the fly."""
        sources = self._retriever.search(query, k)
        context = self._build_context(sources)
        system_prompt = (
            "You are a helpful assistant for the vLLM codebase. "
            "Answer the question concisely and truthfully based ONLY "
            "on the provided context."
        )
        user_prompt = f"Context:\n{context}\n\nQuestion: {query}"
        return self._llm.generate(system_prompt, user_prompt)

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
            context = self._build_context(item.retrieved_sources)
            system_prompt = (
                "You are a helpful assistant for the vLLM codebase. "
                "Answer the question concisely based ONLY on the "
                "provided context."
            )
            user_prompt = f"Context:\n{context}\n\nQuestion: {item.question}"
            ans_text = self._llm.generate(system_prompt, user_prompt)

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
