from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A segment of a document with exact character coordinates."""

    file_path: str
    content: str
    first_character_index: int
    last_character_index: int

    def __str__(self) -> str:
        res = ""
        res += (
            f"chunk: {self.file_path}"
            f"[{self.first_character_index}:"
            f"{self.last_character_index}]\n"
        )
        res += self.content
        return res


@dataclass(frozen=True)
class Zone:
    """A continuous character span representing a semantic zone."""

    start_char: int
    end_char: int
