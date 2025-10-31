"""OpenAI API implementation."""

import logging
from collections.abc import Generator

from src.config import get_config
from src.llm.base import LLMInterface

logger = logging.getLogger(__name__)


class OpenAILLM(LLMInterface):
    """OpenAI API implementation."""

    def __init__(self, config=None):
        """Initialize OpenAI LLM.

        Args:
            config: Configuration instance (creates new if None)
        """
        self.config = config or get_config()

        if not self.config.openai_api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        # Import openai only when needed
        try:
            from openai import OpenAI

            self.client = OpenAI(api_key=self.config.openai_api_key)
            logger.info(f"Initialized OpenAI client with model: {self.config.openai_model_name}")
        except ImportError:
            raise ImportError("OpenAI package not installed. Install with: pip install openai")

    def generate(
        self, prompt: str, system_prompt: str | None = None, stream: bool = False
    ) -> Generator[str]:
        """Generate response from OpenAI API.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            stream: Whether to stream the response

        Returns:
            Generator that yields text chunks
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        # GPT-5 models use max_completion_tokens instead of max_tokens
        is_gpt5_model = self.config.openai_model_name.startswith("gpt-5")

        try:
            if stream and not is_gpt5_model:
                # Streaming response (GPT-5 models require verified org for streaming)
                response = self.client.chat.completions.create(
                    model=self.config.openai_model_name,
                    messages=messages,
                    max_tokens=self.config.openai_max_tokens,
                    temperature=self.config.openai_temperature,
                    stream=True,
                )

                for chunk in response:
                    if chunk.choices[0].delta.content is not None:
                        yield chunk.choices[0].delta.content
            elif stream and is_gpt5_model:
                # GPT-5 models: streaming requires verified org, so fall back to non-streaming
                # and simulate streaming by yielding the full response
                logger.warning(
                    "GPT-5 streaming requires verified organization. Using non-streaming mode."
                )
                response = self.client.chat.completions.create(
                    model=self.config.openai_model_name,
                    messages=messages,
                    max_completion_tokens=self.config.openai_max_tokens,
                )
                content = response.choices[0].message.content
                if content:
                    yield content
            else:
                # Non-streaming response - yield as single chunk
                if is_gpt5_model:
                    # GPT-5 models don't support custom temperature, only use max_completion_tokens
                    response = self.client.chat.completions.create(
                        model=self.config.openai_model_name,
                        messages=messages,
                        max_completion_tokens=self.config.openai_max_tokens,
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.config.openai_model_name,
                        messages=messages,
                        max_tokens=self.config.openai_max_tokens,
                        temperature=self.config.openai_temperature,
                    )

                content = response.choices[0].message.content
                yield content.strip() if content else ""

        except Exception as e:
            logger.error(f"OpenAI API error: {e}")
            raise

    def __repr__(self) -> str:
        """String representation of the LLM."""
        return f"OpenAILLM(model={self.config.openai_model_name})"
