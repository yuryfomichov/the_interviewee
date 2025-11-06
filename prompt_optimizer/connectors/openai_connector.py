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
        """Test via OpenAI API (async).

        Args:
            system_prompt: System prompt to use
            message: User message

        Returns:
            Model response
        """
        try:
            # Using the async API
            response = await self.client.chat.completions.with_raw_response.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": message},
                ],
                temperature=0.7,
            )
            # Parse the response
            completion = response.parse()
            return completion.choices[0].message.content or ""
        except Exception as e:
            logger.error(f"OpenAI API call failed: {e}")
            raise
