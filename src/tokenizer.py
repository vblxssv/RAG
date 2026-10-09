import re


class CodeTokenizer:
    """Tokenize code identifiers and text tokens."""

    def __init__(self, min_token_len: int = 2) -> None:
        self._min_token_len = min_token_len
        self._camel_regex = re.compile(r"([a-z])([A-Z])")
        self._word_regex = re.compile(r"[a-zA-Z0-9_]+")

    def tokenize(self, text: str) -> list[str]:
        if not text:
            return []

        spaced = self._camel_regex.sub(r"\1 \2", text)

        tokens: list[str] = []
        for word in self._word_regex.findall(spaced):
            for part in word.split("_"):
                part_clean = part.lower()
                if len(part_clean) >= self._min_token_len:
                    tokens.append(part_clean)

        return tokens
