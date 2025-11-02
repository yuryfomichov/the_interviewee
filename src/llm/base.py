"""Base interface for LLM implementations."""

from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator
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

    config: "Config"
    retriever: "BaseRetriever"
    user_name: str
    last_prompt: str | None
    system_prompt_fn: Callable[[str], str]
    system_prompt: str

    def __init__(
        self,
        config: "Config",
        retriever: "BaseRetriever",
        user_name: str,
        system_prompt_fn: Callable[[str], str],
    ) -> None:
        """Common initialization for concrete LLMs."""
        self.config = config
        self.retriever = retriever
        self.user_name = user_name
        self.last_prompt = None
        self.system_prompt_fn = system_prompt_fn
        self.system_prompt = system_prompt_fn(user_name)

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
