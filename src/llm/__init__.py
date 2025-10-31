"""LLM interface abstraction for multiple backends (MLX, OpenAI)."""

from src.llm.base import LLMInterface
from src.llm.factory import create_llm
from src.llm.mlx_llm import MLXLocalLLM
from src.llm.openai_llm import OpenAILLM

__all__ = [
    "LLMInterface",
    "MLXLocalLLM",
    "OpenAILLM",
    "create_llm",
]
