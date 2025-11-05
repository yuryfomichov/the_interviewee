"""Tools for testing target models with different prompts."""

import logging

from prompt_optimizer.model_clients import ModelClient

logger = logging.getLogger(__name__)


def test_target_model(system_prompt: str, message: str, model_client: ModelClient) -> str:
    """
    Test the target model with a system prompt and message.

    Args:
        system_prompt: System prompt to test
        message: User message to send
        model_client: Client for the target model

    Returns:
        Model's response string
    """
    logger.debug(f"Testing prompt with message: {message[:50]}...")
    response = model_client.test_prompt(system_prompt, message)
    logger.debug(f"Received response: {response[:100]}...")
    return response
