from dataclasses import dataclass

from pydantic import BaseModel


class MinimalSource(BaseModel):
    """A segment of a document with exact character coordinates."""

    file_path: str
    first_character_index: int
    last_character_index: int


@dataclass(frozen=True)
class Zone:
    """A continuous character span representing a semantic zone."""

    start_char: int
    end_char: int
