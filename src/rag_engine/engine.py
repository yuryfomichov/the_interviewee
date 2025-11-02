"""RAG (Retrieval-Augmented Generation) engine for AI Interviewee."""

import logging
from collections.abc import Iterator

from langchain_community.chat_message_histories import ChatMessageHistory

from src.config import Config, get_config
from src.llm.base import LLMInputs, LLMInterface
from src.prompts import OUT_OF_SCOPE_RESPONSE
from src.rag_engine.prompt_logger import PromptLogger

logger = logging.getLogger(__name__)


class RAGEngine:
    """Retrieval-Augmented Generation engine for interview responses."""

    def __init__(
        self,
        llm: LLMInterface,
        config: Config | None = None,
    ):
        """Initialize RAG engine with injected dependencies.

        Args:
            llm: LLM instance (must implement invoke and stream methods)
            config: Configuration instance (creates new if None)
        """
        self.config = config or get_config()
        self.llm = llm

        # Create conversational memory
        self.memory = ChatMessageHistory()
        self.prompt_logger = PromptLogger()

        logger.info(f"RAG engine initialized with LLM: {type(llm).__name__}")

    def _is_out_of_scope(self, question: str) -> bool:
        """Check if question is out of scope (basic heuristic).

        Args:
            question: User question

        Returns:
            True if question appears out of scope
        """
        # Keywords that might indicate out-of-scope questions
        out_of_scope_keywords = [
            "personal life",
            "family",
            "religion",
            "political",
            "politics",
            "confidential",
            "other candidates",
            "salary history",
            "marital status",
        ]

        question_lower = question.lower()

        for keyword in out_of_scope_keywords:
            if keyword in question_lower:
                logger.info(f"Question flagged as potentially out-of-scope: {keyword}")
                return True

        return False

    def generate_response(
        self, question: str, use_history: bool = True, stream: bool = False
    ) -> Iterator[str]:
        """Generate response to interview question.

        Args:
            question: User question
            use_history: Whether to consider conversation history
            stream: Whether to stream the response

        Yields:
            Response tokens
        """
        logger.info(f"Generating response for: {question[:100]}...")

        # Check if question is out of scope
        if self._is_out_of_scope(question):
            yield OUT_OF_SCOPE_RESPONSE
            return

        try:
            # Get chat history if needed
            chat_history = self.memory.messages if use_history else []

            # Prepare inputs
            inputs = LLMInputs(question=question, chat_history=chat_history)

            # Collect full response for history
            full_response = ""

            if stream:
                # Streaming - manually manage history
                for chunk in self.llm.stream(inputs):
                    # chunk is a string token
                    token = chunk if isinstance(chunk, str) else str(chunk)
                    full_response += token
                    yield token
                logger.info("Streaming response generated successfully")
            else:
                # Non-streaming - manually manage history
                result = self.llm.invoke(inputs)
                response = result if isinstance(result, str) else str(result)
                full_response = response
                yield response
                logger.info("Response generated successfully")

            # Manually add to history
            from langchain_core.messages import AIMessage, HumanMessage

            self.memory.add_message(HumanMessage(content=question))
            self.memory.add_message(AIMessage(content=full_response))
            self.prompt_logger.log(question, self.llm.last_prompt, full_response)

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            error_msg = (
                "I apologize, but I encountered an error. "
                "Please try again or rephrase your question."
            )
            yield error_msg

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.memory.clear()
        logger.info("Conversation history cleared")

    def __repr__(self) -> str:
        """String representation of RAG engine."""
        return f"RAGEngine(llm={type(self.llm).__name__})"
