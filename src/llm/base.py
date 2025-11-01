"""Base interface for LLM implementations."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from typing import TYPE_CHECKING

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from langchain_core.retrievers import BaseRetriever

    from src.config import Config


class LLMInputs(BaseModel):
    """Input model for LLM invoke and stream methods."""

    model_config = {"arbitrary_types_allowed": True}

    question: str = Field(..., description="The user's question")
    chat_history: list[BaseMessage] = Field(
        default_factory=list, description="Chat message history (LangChain message objects)"
    )


class LLMInterface(ABC):
    """Abstract base class for LLM implementations.

    LLM implementations must support RAG-based invoke and stream methods.
    """

    @abstractmethod
    def initialize(self, config: "Config", retriever: "BaseRetriever", user_name: str) -> None:
        """Initialize the LLM with config, retriever and user name.

        Must be called before invoke() or stream().

        Args:
            config: Configuration instance
            retriever: Vector store retriever for RAG
            user_name: Name of the user/candidate
        """
        pass

    @abstractmethod
    def invoke(self, inputs: LLMInputs) -> str:
        """Non-streaming invocation.

        Args:
            inputs: LLM inputs containing question and chat_history

        Returns:
            Generated response as a string
        """
        pass

    @abstractmethod
    def stream(self, inputs: LLMInputs) -> Iterator[str]:
        """Streaming invocation.

        Args:
            inputs: LLM inputs containing question and chat_history

        Yields:
            Response tokens/chunks as strings
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the LLM."""
        pass
