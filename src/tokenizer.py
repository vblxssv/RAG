"""Code and natural language tokenizer for lexical indexing and search."""

import re


class CodeTokenizer:
    """Tokenizes code identifiers and natural language text."""

    def __init__(self, min_token_len: int = 2) -> None:
        """Initialize regex splitters and minimum token length threshold."""
        self._min_token_len = min_token_len
        self._acronym_regex = re.compile(r"([A-Z]+)([A-Z][a-z])")
        self._camel_regex = re.compile(r"([a-z\d])([A-Z])")
        self._word_regex = re.compile(r"[a-zA-Z0-9_]+")

    def tokenize(self, text: str) -> list[str]:
        """Split text into lowercase alphanumeric tokens."""
        if not text:
            return []

        tokens: list[str] = []
        for word in self._word_regex.findall(text):
            parts = word.split("_")
            has_split = len(parts) > 1
            word_lower = word.lower()
            seen_in_word: set[str] = set()

            for part in parts:
                if not part:
                    continue
                spaced = self._acronym_regex.sub(r"\1 \2", part)
                spaced = self._camel_regex.sub(r"\1 \2", spaced)
                sub_tokens = spaced.split()
                if len(sub_tokens) > 1:
                    has_split = True
                for sub in sub_tokens:
                    sub_clean = sub.lower()
                    if len(sub_clean) >= self._min_token_len:
                        tokens.append(sub_clean)
                        seen_in_word.add(sub_clean)

            if (
                has_split
                and len(word_lower) >= self._min_token_len
                and word_lower not in seen_in_word
            ):
                tokens.append(word_lower)

        return tokens
