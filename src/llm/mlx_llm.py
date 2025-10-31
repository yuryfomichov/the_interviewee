"""Apple Silicon optimized LLM using MLX."""

import logging
from collections.abc import Generator

from mlx_lm.generate import generate, stream_generate
from mlx_lm.utils import load

from src.config import get_config
from src.llm.base import LLMInterface

logger = logging.getLogger(__name__)


class MLXLocalLLM(LLMInterface):
    """Apple Silicon optimized LLM using MLX with quantization support."""

    def __init__(self, config=None):
        """Initialize MLX LLM.

        Args:
            config: Configuration instance (creates new if None)
        """
        self.config = config or get_config()

        # Load model using MLX with automatic quantization
        model_name = self.config.local_model_name

        logger.info(f"Loading model with MLX: {model_name}")
        logger.info("Device: Apple Silicon (Metal)")

        # Prepare authentication token for gated models (e.g., Llama)
        token = self.config.huggingface_token
        if token:
            logger.info("Using HuggingFace authentication token")

        try:
            # MLX load function does not accept quantization parameter
            # Quantization is handled by loading pre-quantized models from HuggingFace
            # or by converting models separately using mlx_lm.convert
            import os

            if token:
                os.environ["HF_TOKEN"] = token

            # Load model and tokenizer (don't need config)
            model, tokenizer = load(model_name, return_config=False)  # type: ignore[misc]
            self.model = model
            self.tokenizer = tokenizer
            logger.info("Model loaded successfully with MLX")
        except Exception as e:
            logger.error(f"Failed to load model with MLX: {e}")
            raise

    def generate(
        self, prompt: str, system_prompt: str | None = None, stream: bool = False
    ) -> Generator[str]:
        """Generate response using MLX.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            stream: Whether to stream the response (yields tokens)

        Returns:
            Generator that yields text chunks
        """
        # Format prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
        else:
            full_prompt = prompt

        # Generate with MLX
        try:
            if stream:
                has_yielded = False
                try:
                    for response in stream_generate(
                        self.model,
                        self.tokenizer,
                        prompt=full_prompt,
                        max_tokens=self.config.local_model_max_tokens,
                    ):
                        # stream_generate returns a GenerationResponse object
                        # with a 'text' attribute containing the generated text
                        if hasattr(response, "text"):
                            text = response.text
                        else:
                            # Fallback: response might be a tuple (text, _)
                            text = response[0] if isinstance(response, tuple) else str(response)

                        has_yielded = True
                        yield text
                except StopIteration:
                    pass

                # If nothing was yielded, ensure we yield at least something
                if not has_yielded:
                    logger.warning("Stream generated no tokens, falling back to non-streaming")
                    response = generate(
                        self.model,
                        self.tokenizer,
                        prompt=full_prompt,
                        max_tokens=self.config.local_model_max_tokens,
                        verbose=False,
                    )
                    yield response.strip()
            else:
                # Non-streaming: yield complete response as single chunk
                response = generate(
                    self.model,
                    self.tokenizer,
                    prompt=full_prompt,
                    max_tokens=self.config.local_model_max_tokens,
                    verbose=False,
                )
                yield response.strip()
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise

    def __repr__(self) -> str:
        """String representation of the LLM."""
        return f"MLXLocalLLM(model={self.config.local_model_name}, device=Apple Silicon)"
