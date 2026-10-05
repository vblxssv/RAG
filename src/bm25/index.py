from collections import Counter, defaultdict
import math
from typing import Dict, List, Self, Tuple, Any


class BM25Indexer:
    def __init__(self, inverted_index: Dict[str, List[Tuple[int, int]]],
                 word_value: Dict[str, float], chunk_lens: List[int],
                 av_chunk_len: float,
                 k1: float = 1.5, b: float = 0.75
                 ) -> None:
        self._inverted_index = inverted_index
        self._word_value = word_value
        self._chunk_lens = chunk_lens
        self._av_chunk_len = av_chunk_len
        self._k1 = k1
        self._b = b

    def __str__(self) -> str:
        lines = [
            f"BM25Indexer (k1={self._k1}, b={self._b}, "
            f"avg_chunk_len={self._av_chunk_len:.2f}):",
            f"  Chunk lengths ({len(self._chunk_lens)} docs): "
            f"{self._chunk_lens}",
            "  Inverted Index & IDFs:",
        ]
        for term, postings in self._inverted_index.items():
            idf = self._word_value.get(term, 0.0)
            postings_str = ", ".join(
                f"(doc={doc}, tf={tf})" for doc, tf in postings
            )
            lines.append(f"    '{term}' (idf={idf:.4f}): [{postings_str}]")
        return "\n".join(lines)

    @classmethod
    def build(cls, tokens: List[List[str]],
              k1: float = 1.5, b: float = 0.75) -> Self:
        total_docs = len(tokens)
        if not total_docs:
            return cls({}, {}, [], 0.0, k1, b)

        chunk_lens = [len(cur) for cur in tokens]
        av_len = sum(chunk_lens) / len(tokens)

        inverted_index: defaultdict[str, list[tuple[int, int]]] = (
            defaultdict(list)
        )
        for doc_id, doc_tokens in enumerate(tokens):
            term_counts = Counter(doc_tokens)
            for term, count in term_counts.items():
                inverted_index[term].append((doc_id, count))

        idf: dict[str, float] = {}
        for term, posting_list in inverted_index.items():
            n_q = len(posting_list)
            idf[term] = math.log(1.0 + (total_docs - n_q + 0.5) / (n_q + 0.5))

        return cls(dict(inverted_index), idf, chunk_lens, av_len, k1, b)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "inverted_index": self._inverted_index,
            "word_value": self._word_value,
            "chunk_lens": self._chunk_lens,
            "av_chunk_len": self._av_chunk_len,
            "k1": self._k1,
            "b": self._b,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Self:
        inverted_index = {
            term: [(doc, tf) for doc, tf in postings]
            for term, postings in data["inverted_index"].items()
        }
        return cls(
            inverted_index=inverted_index,
            word_value=data["word_value"],
            chunk_lens=data["chunk_lens"],
            av_chunk_len=data["av_chunk_len"],
            k1=data.get("k1", 1.5),
            b=data.get("b", 0.75),
        )
