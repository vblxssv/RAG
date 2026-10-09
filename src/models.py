import uuid
from typing import List, Union
from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int

    def __str__(self) -> str:
        res = ''
        res += f"{self.file_path} "
        res += f"[{self.first_character_index}:{self.last_character_index}]"
        return res

    @property
    def interval_len(self) -> int:
        return self.last_character_index - self.first_character_index

    def _get_intersection(self, other: MinimalSource) -> int:
        if self.file_path != other.file_path:
            return 0
        start_intersec = max(self.first_character_index,
                             other.first_character_index)
        end_intersec = min(self.last_character_index,
                           other.last_character_index)
        return max(0, end_intersec - start_intersec)

    def get_iou(self, other: MinimalSource) -> float:
        if self.file_path != other.file_path:
            return 0.0
        intersection = self._get_intersection(other)
        union = self.interval_len + other.interval_len - intersection
        if union <= 0:
            return 0.0
        return intersection / union

# ================================================


class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]

# ================================================


class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int

# ================================================


class MinimalAnswer(MinimalSearchResults):
    answer: str


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int

# ================================================
