"""Pydantic data models for the RAG pipeline."""

import uuid
from typing import List, Union
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Represents a source text slice location in a codebase file."""

    file_path: str
    first_character_index: int
    last_character_index: int

    def __str__(self) -> str:
        """Return formatted file path and character range."""
        return (
            f"{self.file_path} "
            f"[{self.first_character_index}:{self.last_character_index}]"
        )

    @property
    def interval_len(self) -> int:
        """Return character length of the source span."""
        return self.last_character_index - self.first_character_index

    def _get_intersection(self, other: "MinimalSource") -> int:
        """Compute character overlap length between two source spans."""
        if self.file_path != other.file_path:
            return 0
        start_intersec = max(
            self.first_character_index, other.first_character_index
        )
        end_intersec = min(
            self.last_character_index, other.last_character_index
        )
        return max(0, end_intersec - start_intersec)

    def get_iou(self, other: "MinimalSource") -> float:
        """Compute Intersection over Union between two source spans."""
        if self.file_path != other.file_path:
            return 0.0
        intersection = self._get_intersection(other)
        union = self.interval_len + other.interval_len - intersection
        if union <= 0:
            return 0.0
        return intersection / union


class UnansweredQuestion(BaseModel):
    """Represents an unanswered question."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Represents an answered question with ground truth sources."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Represents a dataset of questions."""

    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    """Search results containing retrieved sources for a question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class StudentSearchResults(BaseModel):
    """Batch search results container across questions."""

    search_results: List[MinimalSearchResults]
    k: int


class MinimalAnswer(MinimalSearchResults):
    """Search results with generated natural language answer."""

    answer: str


class StudentSearchResultsAndAnswer(BaseModel):
    """Batch search results container with generated answers."""

    search_results: List[MinimalAnswer]
    k: int
