"""Dummy connector for testing - provides fast, deterministic responses."""

from prompt_optimizer.connectors.base import BaseConnector


class DummyConnector(BaseConnector):
    """A dummy connector that returns simple, fixed responses for testing."""

    async def test_prompt(self, system_prompt: str, message: str) -> str:
        """Return a simple fixed response."""
        return "test response"
