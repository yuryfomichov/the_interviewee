"""Factory function for creating LLM instances."""

import logging
import platform
from collections.abc import Callable
from typing import TYPE_CHECKING, Literal, TypedDict

from src.config import Config, get_config
from src.llm.base import LLMInterface
from src.prompts import get_default_system_prompt

if TYPE_CHECKING:
    from langchain_core.retrievers import BaseRetriever


class ProviderConfig(TypedDict):
    backend: Literal["mlx", "openai"]
    system_prompt_fn: Callable[[str], str]


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

# Provider configuration mapping
PROVIDER_CONFIG: dict[str, ProviderConfig] = {
    "qwen": {
        "backend": "mlx",
        "system_prompt_fn": get_default_system_prompt,
    },
    "llama": {
        "backend": "mlx",
        "system_prompt_fn": get_default_system_prompt,
    },
    "openai": {
        "backend": "openai",
        "system_prompt_fn": get_default_system_prompt,
    },
}


def create_llm(
    *,
    retriever: "BaseRetriever",
    config: Config | None = None,
    user_name: str | None = None,
) -> LLMInterface:
    """Create and fully initialize the appropriate LLM based on the current platform."""
    if retriever is None:
        raise ValueError("retriever must be provided to create an LLM instance.")

    config = config or get_config()
    user_name = user_name or config.user_name
    provider = config.model_provider.lower()

    # Check if provider is valid
    if provider not in PROVIDER_CONFIG:
        valid_providers = ", ".join(f"'{p}'" for p in PROVIDER_CONFIG.keys())
        raise ValueError(f"Unknown model provider: {provider}. Choose from: {valid_providers}")

    provider_config = PROVIDER_CONFIG[provider]
    backend = provider_config["backend"]
    system_prompt_fn = provider_config["system_prompt_fn"]

    if backend == "mlx":
        # Use MLX for Apple Silicon (qwen, llama)
        if IS_APPLE_SILICON and MLX_AVAILABLE:
            logger.info(f"Creating {provider} LLM with MLX (Apple Silicon optimized)")
            from src.llm.mlx_llm import MLXLocalLLM

            return MLXLocalLLM(
                config=config,
                retriever=retriever,
                user_name=user_name,
                system_prompt_fn=system_prompt_fn,
            )
        else:
            raise ImportError(
                f"{provider} models require MLX on Apple Silicon. Install with: pip install mlx mlx-lm"
            )
    elif backend == "openai":
        logger.info("Creating OpenAI LLM")
        from src.llm.openai_llm import OpenAILLM

        return OpenAILLM(
            config=config,
            retriever=retriever,
            user_name=user_name,
            system_prompt_fn=system_prompt_fn,
        )
    else:
        raise ValueError(f"Unknown backend: {backend}")
