"""Recall calculation module for retrieval evaluation."""

from dataclasses import dataclass
from pathlib import Path
from typing import List
from src.storage import RetrievingStorage
from src.models import MinimalSource, AnsweredQuestion


@dataclass
class EvalPair:
    """Pair of ground truth and retrieved sources for a single question."""

    question_id: str
    gt_sources: List[MinimalSource]
    my_sources: List[MinimalSource]

    def recall_at_k(self, k: int) -> float:
        """Calculate recall@k for this single question."""
        if not self.gt_sources:
            return 1.0
        top_k = self.my_sources[:k]
        hits = 0
        for gt in self.gt_sources:
            for my in top_k:
                if gt.get_iou(my) > 0.05:
                    hits += 1
                    break
        return hits / len(self.gt_sources)


class Recall:
    """Evaluates retrieval quality using Recall@k over ground truth."""

    def __init__(
        self,
        search_result_path: str | Path,
        dataset_path: str | Path,
    ) -> None:
        """Initialize with search results and ground truth dataset."""
        storage = RetrievingStorage(
            dataset_path,
            Path(search_result_path).parent,
        )
        self._standart = storage.load_dataset()
        self._results = storage.load_search_results(search_result_path)

        gt_map = {
            q.question_id: q.sources
            for q in self._standart.rag_questions
            if isinstance(q, AnsweredQuestion)
        }
        self._pairs: List[EvalPair] = []
        for res in self._results.search_results:
            if res.question_id in gt_map:
                pair = EvalPair(
                    question_id=res.question_id,
                    gt_sources=gt_map[res.question_id],
                    my_sources=res.retrieved_sources,
                )
                self._pairs.append(pair)

    def calculate_at_k(self, k: int) -> float:
        """Calculate average Recall@k across all evaluated question pairs."""
        if not self._pairs:
            return 0.0
        return sum(p.recall_at_k(k) for p in self._pairs) / len(self._pairs)
