"""Function-based model client for in-process testing."""

import logging
from collections.abc import Callable

logger = logging.getLogger(__name__)


class FunctionModelClient:
    """Model client that uses a Python function."""

    def __init__(self, func: Callable[[str, str], str]):
        """
        Initialize with a callable function.

        Args:
            func: Function that takes (system_prompt: str, message: str) -> response: str
        """
        self.func = func
        logger.info("FunctionModelClient initialized")

    def test_prompt(self, system_prompt: str, message: str) -> str:
        """Test via function call."""
        return self.func(system_prompt, message)
