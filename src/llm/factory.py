"""Factory function for creating LLM instances."""

import logging
import platform

from src.config import get_config
from src.llm.base import LLMInterface

# Detect if we're on Apple Silicon
IS_APPLE_SILICON = platform.system() == "Darwin" and platform.machine() == "arm64"

# Check available backends
MLX_AVAILABLE = False
if IS_APPLE_SILICON:
    try:
        import mlx.core as mx  # noqa: F401
        import mlx_lm  # noqa: F401

        MLX_AVAILABLE = True
    except ImportError:
        pass

logger = logging.getLogger(__name__)


def create_llm(config=None, retriever=None, user_name: str = "") -> LLMInterface:
    """Factory function to create appropriate LLM based on configuration and platform.

    Args:
        config: Configuration instance (creates new if None)
        retriever: Vector store retriever for RAG
        user_name: Name of the user/candidate

    Returns:
        LLM interface instance
    """
    config = config or get_config()

    provider = config.model_provider.lower()

    if provider == "local":
        # Use MLX for Apple Silicon
        if IS_APPLE_SILICON and MLX_AVAILABLE:
            logger.info("Creating local LLM with MLX (Apple Silicon optimized)")
            from src.llm.mlx_llm import MLXLocalLLM

            return MLXLocalLLM(config=config, retriever=retriever, user_name=user_name)
        else:
            raise ImportError(
                "Local models require MLX on Apple Silicon. Install with: pip install mlx mlx-lm"
            )
    elif provider == "openai":
        logger.info("Creating OpenAI LLM")
        from src.llm.openai_llm import OpenAILLM

        return OpenAILLM(config=config, retriever=retriever, user_name=user_name)
    else:
        raise ValueError(f"Unknown model provider: {provider}. Choose 'local' or 'openai'.")
