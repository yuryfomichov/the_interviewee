"""Apple Silicon optimized LLM using MLX."""

import logging
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, cast

import mlx.nn as nn
from langchain_core.retrievers import BaseRetriever
from mlx_lm.generate import stream_generate
from mlx_lm.tokenizer_utils import TokenizerWrapper
from mlx_lm.utils import load as mlx_load

from src.llm.base import LLMInputs, LLMInterface
from src.prompts import format_rag_prompt

if TYPE_CHECKING:
    from src.config import Config

from src.config import MLXModelSettings

logger = logging.getLogger(__name__)


class MLXLocalLLM(LLMInterface):
    """Apple Silicon optimized LLM using MLX with LangChain integration."""

    model: nn.Module
    tokenizer: TokenizerWrapper
    settings: MLXModelSettings

    def __init__(
        self,
        config: "Config",
        retriever: BaseRetriever,
        user_name: str,
        system_prompt_fn: Callable[[str], str],
    ):
        """Instantiate the MLX-backed LLM with all required dependencies."""
        super().__init__(
            config=config,
            retriever=retriever,
            user_name=user_name,
            system_prompt_fn=system_prompt_fn,
        )

        # Get model settings
        model_settings = self.config.get_model_settings()
        if not isinstance(model_settings, MLXModelSettings):
            raise TypeError("MLXLocalLLM requires MLX model settings.")
        self.settings = cast(MLXModelSettings, model_settings)

        model_name = self.config.get_model_name()
        logger.info(f"Loading model with MLX: {model_name}")
        logger.info("Device: Apple Silicon (Metal)")

        # Prepare authentication token for gated models
        token = self.config.huggingface_token
        if token:
            logger.info("Using HuggingFace authentication token")
            import os

            os.environ["HF_TOKEN"] = token

        try:
            # Load model and tokenizer using mlx_lm directly
            logger.info("Loading model with mlx_lm...")
            model_artifacts = mlx_load(model_name)
            # mlx_load can return 2 or 3 items (model, tokenizer, [config])
            self.model = model_artifacts[0]
            self.tokenizer = model_artifacts[1]

            # Ensure we have a chat template for instruct-tuned models
            if (
                hasattr(self.tokenizer, "default_chat_template")
                and getattr(self.tokenizer, "chat_template", None) is None
            ):
                self.tokenizer.chat_template = self.tokenizer.default_chat_template
            # Ensure tokenizer stops on modern special tokens
            add_eos_token = getattr(self.tokenizer, "add_eos_token", None)
            if callable(add_eos_token):
                for eos_token in ("<|eot_id|>", "<|end_of_text|>", "</s>"):
                    try:
                        add_eos_token(eos_token)
                    except Exception:
                        continue

            logger.info("MLX model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def _format_docs(self, docs):
        """Format retrieved documents into context string."""
        if not docs:
            return "No specific career information retrieved."
        context_parts = []
        for i, doc in enumerate(docs, 1):
            # Handle both Document objects and strings
            if isinstance(doc, str):
                context_parts.append(f"[Source {i}]\n{doc}\n")
            else:
                # Document object with metadata
                source = (
                    doc.metadata.get("source", "Unknown") if hasattr(doc, "metadata") else "Unknown"
                )
                source_name = source.split("/")[-1] if "/" in source else source
                page_content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                context_parts.append(f"[Source {i}: {source_name}]\n{page_content}\n")
        return "\n".join(context_parts)

    def _normalize_history(self, chat_history: list) -> list[dict]:
        """Convert LangChain chat messages into tokenizer-compatible dicts."""
        normalized: list[dict] = []
        for msg in chat_history or []:
            content = msg.content if hasattr(msg, "content") else str(msg)
            if not content:
                continue

            role = getattr(msg, "role", None)
            if role not in {"user", "assistant", "system"}:
                role = "assistant" if getattr(msg, "type", "") == "ai" else "user"
            normalized.append({"role": role, "content": content})
        return normalized

    def _build_chat_prompt(self, question: str, context: str, chat_history: list) -> str:
        """Build prompt using tokenizer chat template."""
        system_sections = [self.system_prompt.strip()]
        if context.strip():
            system_sections.append("Use these career details to answer specifically:")
            system_sections.append(context.strip())
        if chat_history:
            system_sections.append(
                "Conversation history follows for context only. Do not repeat its format."
            )

        messages: list[dict] = [{"role": "system", "content": "\n\n".join(system_sections)}]
        messages.extend(self._normalize_history(chat_history))
        messages.append({"role": "user", "content": question.strip()})

        template_fn = getattr(self.tokenizer, "apply_chat_template", None)
        if not callable(template_fn):
            raise AttributeError("Tokenizer does not support chat templates.")

        prompt = template_fn(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return cast(str, prompt)

    def _build_text_prompt(self, question: str, context: str, chat_history: list) -> str:
        """Fallback prompt builder using plain text template."""
        history_str = ""
        if chat_history:
            for msg in chat_history:
                role = "Human" if hasattr(msg, "type") and msg.type == "human" else "AI"
                content = msg.content if hasattr(msg, "content") else str(msg)
                history_str += f"{role}: {content}\n"

        return format_rag_prompt(
            system_prompt=self.system_prompt,
            context=context,
            chat_history=history_str,
            question=question,
        )

    def _build_prompt(self, inputs: LLMInputs) -> str:
        """Build prompt string, preferring chat templates when available."""
        question = inputs.question
        chat_history = inputs.chat_history

        docs = self.retriever.invoke(question)
        context = self._format_docs(docs)

        prompt: str | None = None
        template_fn = getattr(self.tokenizer, "apply_chat_template", None)
        has_template = bool(getattr(self.tokenizer, "chat_template", None))
        if callable(template_fn) and has_template:
            try:
                prompt = self._build_chat_prompt(question, context, chat_history)
            except Exception as exc:
                logger.warning(f"Falling back to text prompt due to chat template error: {exc}")

        if prompt is None:
            prompt = self._build_text_prompt(question, context, chat_history)

        self.last_prompt = prompt
        return prompt

    def invoke(self, inputs: LLMInputs) -> str:
        prompt_input = self._build_prompt(inputs)

        # Generate response (non-streaming) using mlx_lm directly
        from mlx_lm.generate import generate as mlx_generate

        response = mlx_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt_input,
            max_tokens=self.settings.max_new_tokens,
            verbose=False,
        )
        return response

    def stream(self, inputs: LLMInputs) -> Iterator[str]:
        prompt_input = self._build_prompt(inputs)

        # Stream response using mlx_lm directly (MLXPipeline streaming is broken)
        for generation_response in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=prompt_input,
            max_tokens=self.settings.max_new_tokens,
        ):
            # stream_generate yields GenerationResponse objects with a text attribute
            yield generation_response.text

    def __repr__(self) -> str:
        """String representation of the LLM."""
        return f"MLXLocalLLM(model={self.config.get_model_name()}, device=Apple Silicon)"
