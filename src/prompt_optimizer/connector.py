"""RAG engine connector for prompt optimization.

This module provides the integration between the prompt optimizer
and the RAG engine, implementing a connector that can be used to
evaluate different system prompts.
"""

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor

from prompt_optimizer import BaseConnector
from src.config import get_config
from src.rag_engine.factory import create_rag_engine

logger = logging.getLogger(__name__)


class RAGConnector(BaseConnector):
    """Connector for testing prompts with the RAG engine."""

    def __init__(self):
        """Initialize RAG connector with the configured RAG engine."""
        config = get_config()
        self.rag_engine = create_rag_engine(config=config)
        self._executor = ThreadPoolExecutor(max_workers=1)
        logger.info(f"RAGConnector initialized with {type(self.rag_engine.llm).__name__}")

    async def test_prompt(self, system_prompt: str, message: str) -> str:
        """Test via RAG engine (async).

        Args:
            system_prompt: System prompt to test
            message: User message

        Returns:
            Model's response
        """
        # Run the synchronous RAG engine call in a thread pool to avoid blocking
        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            self._executor, self._generate_response_sync, system_prompt, message
        )
        return response

    def _generate_response_sync(self, system_prompt: str, message: str) -> str:
        """Synchronous helper to generate response."""
        response_iter = self.rag_engine.generate_response(
            question=message,
            use_history=False,  # Don't use history for testing
            stream=False,
            system_prompt_override=system_prompt,
        )
        # Collect full response
        return "".join(response_iter)
