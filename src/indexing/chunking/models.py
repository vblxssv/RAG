from dataclasses import dataclass

from src.models import MinimalSource


__all__ = ["MinimalSource", "Zone"]


@dataclass(frozen=True)
class Zone:
    """A continuous character span representing a semantic zone."""

    start_char: int
    end_char: int

    @property
    def length(self) -> int:
        """Returns the character length of the zone."""
        return self.end_char - self.start_char

    def exceeds(self, max_size: int) -> bool:
        """Checks if the zone length exceeds the maximum allowed size."""
        return self.length > max_size
