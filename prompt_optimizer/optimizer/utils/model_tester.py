"""Tools for testing target models with different prompts."""

import logging

from prompt_optimizer.connectors import BaseConnector

logger = logging.getLogger(__name__)


async def test_target_model(system_prompt: str, message: str, model_client: BaseConnector) -> str:
    """
    Test the target model with a system prompt and message (async).

    Args:
        system_prompt: System prompt to test
        message: User message to send
        model_client: Connector for the target model

    Returns:
        Model's response string
    """
    logger.debug(f"Testing prompt with message: {message[:50]}...")
    response = await model_client.test_prompt(system_prompt, message)
    logger.debug(f"Received response: {response[:100]}...")
    return response
