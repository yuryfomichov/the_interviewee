"""OpenAI connector for prompt optimization."""

import logging

from openai import AsyncOpenAI

from prompt_optimizer.connectors.base import BaseConnector

logger = logging.getLogger(__name__)


class OpenAIConnector(BaseConnector):
    """Connector for testing prompts with OpenAI models."""

    def __init__(self, api_key: str, model: str = "gpt-4o-mini"):
        """Initialize OpenAI connector.

        Args:
            api_key: OpenAI API key
            model: Model to use (default: gpt-4o-mini)
        """
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        logger.info(f"OpenAIConnector initialized with model {model}")

    async def test_prompt(self, system_prompt: str, message: str) -> str:
        """Test via OpenAI Responses API (async).

        Args:
            system_prompt: System prompt to use (passed as 'instructions')
            message: User message (passed as 'input')

        Returns:
            Model response
        """
        try:
            response = await self.client.responses.create(
                model=self.model,
                instructions=system_prompt,
                input=message,
            )
            return response.output_text or ""
        except Exception as e:
            logger.error(f"OpenAI Responses API call failed: {e}")
            raise
