"""Base protocol for model clients."""

from typing import Protocol


class ModelClient(Protocol):
    """Protocol for target model clients."""

    def test_prompt(self, system_prompt: str, message: str) -> str:
        """Test the model with a system prompt and user message."""
        ...
