"""Base interface for LLM implementations."""

from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any


class LLMInterface(ABC):
    """Abstract base class for LLM implementations."""

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
    def create_rag_chain(
        self,
        retriever: Any,
        memory: Any,
        system_prompt: str,
    ) -> Any:
        """Create a RAG chain with memory for this LLM.

        Args:
            retriever: Vector store retriever
            memory: Chat message history
            system_prompt: System prompt template

        Returns:
            LangChain runnable chain
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the LLM."""
        pass
