"""Model clients for testing target models."""

from prompt_optimizer.model_clients.base import ModelClient
from prompt_optimizer.model_clients.function_client import FunctionModelClient

__all__ = [
    "ModelClient",
    "FunctionModelClient",
]
