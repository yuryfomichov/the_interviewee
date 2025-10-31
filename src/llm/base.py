"""Base interface for LLM implementations."""

from abc import ABC, abstractmethod
from collections.abc import Generator


class LLMInterface(ABC):
    """Abstract base class for LLM implementations."""

    @abstractmethod
    def generate(
        self, prompt: str, system_prompt: str | None = None, stream: bool = False
    ) -> Generator[str]:
        """Generate response from the model.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            stream: Whether to stream the response (yields text chunks if True)

        Returns:
            Generator that yields text chunks. For non-streaming, yields complete response as single chunk.
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the LLM."""
        pass
