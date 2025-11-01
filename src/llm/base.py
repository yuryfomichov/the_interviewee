"""Base interface for LLM implementations."""

from abc import ABC, abstractmethod
from collections.abc import Iterator


class LLMInterface(ABC):
    """Abstract base class for LLM implementations.

    LLM implementations must support RAG-based invoke and stream methods.
    """

    @abstractmethod
    def get_system_prompt(self, user_name: str) -> str:
        """Get the system prompt for this LLM.

        Each LLM implementation can customize the system prompt based on
        its specific capabilities and requirements.

        Args:
            user_name: Name of the user/candidate

        Returns:
            System prompt string
        """
        pass

    @abstractmethod
    def invoke(self, inputs: dict) -> str:
        """Non-streaming invocation.

        Args:
            inputs: Dictionary containing question and chat_history

        Returns:
            Generated response as a string
        """
        pass

    @abstractmethod
    def stream(self, inputs: dict) -> Iterator[str]:
        """Streaming invocation.

        Args:
            inputs: Dictionary containing question and chat_history

        Yields:
            Response tokens/chunks as strings
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the LLM."""
        pass
