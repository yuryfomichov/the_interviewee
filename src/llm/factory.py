"""Factory function for creating LLM instances."""

import logging
import platform

from src.config import get_config
from src.llm.base import LLMInterface
from src.prompts import get_default_system_prompt

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
PROVIDER_CONFIG = {
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


def create_llm() -> LLMInterface:
    """Factory function to create appropriate LLM based on platform.

    Returns:
        Uninitialized LLM interface instance (call initialize() before use)
    """
    config = get_config()
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

            return MLXLocalLLM(system_prompt_fn=system_prompt_fn)
        else:
            raise ImportError(
                f"{provider} models require MLX on Apple Silicon. Install with: pip install mlx mlx-lm"
            )
    elif backend == "openai":
        logger.info("Creating OpenAI LLM")
        from src.llm.openai_llm import OpenAILLM

        return OpenAILLM(system_prompt_fn=system_prompt_fn)
    else:
        raise ValueError(f"Unknown backend: {backend}")
