"""RAG engine connector for prompt optimization.

This module provides the integration between the prompt optimizer
and the RAG engine, creating a test function that can be used to
evaluate different system prompts.
"""

import logging
from collections.abc import Callable

from src.config import get_config
from src.rag_engine.factory import create_rag_engine

logger = logging.getLogger(__name__)


def create_rag_test_function() -> Callable[[str, str], str]:
    """
    Create a test function that wraps the RAG engine.

    This function initializes the RAG engine once and returns a callable
    that can test different system prompts with user messages.

    Returns:
        Function that takes (system_prompt: str, message: str) -> response: str
    """
    # Initialize RAG engine (once)
    config = get_config()
    rag_engine = create_rag_engine(config=config)

    logger.info(f"RAG engine initialized with {type(rag_engine.llm).__name__}")

    def test_prompt(system_prompt: str, message: str) -> str:
        """
        Test function for the optimizer.

        Args:
            system_prompt: System prompt to test
            message: User message

        Returns:
            Model's response
        """
        # Generate response with overridden system prompt
        response_iter = rag_engine.generate_response(
            question=message,
            use_history=False,  # Don't use history for testing
            stream=False,
            system_prompt_override=system_prompt,
        )

        # Collect full response
        return "".join(response_iter)

    return test_prompt
