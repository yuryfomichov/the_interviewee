"""Base connector class for target models."""

from abc import ABC, abstractmethod


class BaseConnector(ABC):
    """Base class for connectors that test target models.

    Users should implement this class to create custom connectors
    that integrate their specific models or systems into the
    prompt optimization pipeline.
    """

    @abstractmethod
    async def test_prompt(self, system_prompt: str, message: str) -> str:
        """Test the target model with a system prompt and user message.

        Args:
            system_prompt: The system prompt to use
            message: The user message to send

        Returns:
            The model's response as a string
        """
        ...
