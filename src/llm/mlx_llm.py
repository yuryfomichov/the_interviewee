"""Apple Silicon optimized LLM using MLX."""

import logging
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

from langchain_core.retrievers import BaseRetriever
from mlx_lm.generate import stream_generate
from mlx_lm.utils import load as mlx_load

from src.llm.base import LLMInputs, LLMInterface
from src.prompts import format_rag_prompt

if TYPE_CHECKING:
    import mlx.nn as nn
    from mlx_lm.tokenizer_utils import TokenizerWrapper

    from src.config import Config

logger = logging.getLogger(__name__)


class MLXLocalLLM(LLMInterface):
    """Apple Silicon optimized LLM using MLX with LangChain integration."""

    def __init__(self, system_prompt_fn: Callable[[str], str]):
        """Initialize MLX LLM.

        Args:
            system_prompt_fn: Function that takes user_name and returns system prompt

        Call initialize() before using invoke() or stream().
        """
        self.system_prompt_fn = system_prompt_fn

        # These will be set during initialize()
        self.config: Config
        self.model: nn.Module
        self.tokenizer: TokenizerWrapper
        self.retriever: BaseRetriever
        self.system_prompt: str

    def initialize(self, config, retriever: BaseRetriever, user_name: str) -> None:
        """Initialize the LLM with config, retriever and user name.

        Args:
            config: Configuration instance
            retriever: Vector store retriever for RAG
            user_name: Name of the user/candidate
        """
        self.config = config
        self.retriever = retriever
        self.system_prompt = self.system_prompt_fn(user_name)

        # Get model settings
        self.settings = self.config.get_model_settings()

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

    def _format_prompt(self, question: str, chat_history: list, context: str) -> str:
        """Format the full prompt with system prompt, context, history, and question."""
        if hasattr(self.tokenizer, "apply_chat_template") and getattr(
            self.tokenizer, "chat_template", None
        ):
            system_sections = [self.system_prompt.strip()]
            if context.strip():
                system_sections.append("Use these career details to answer specifically:")
                system_sections.append(context.strip())
            if chat_history:
                system_sections.append(
                    "Conversation history follows for context only. Do not repeat its format."
                )

            messages: list[dict] = [{"role": "system", "content": "\n\n".join(system_sections)}]

            for msg in chat_history or []:
                content = msg.content if hasattr(msg, "content") else str(msg)
                if not content:
                    continue

                role = getattr(msg, "role", None)
                if role not in {"user", "assistant", "system"}:
                    role = "assistant" if getattr(msg, "type", "") == "ai" else "user"

                messages.append({"role": role, "content": content})

            messages.append({"role": "user", "content": question.strip()})

            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception as exc:
                logger.warning(f"Falling back to text prompt due to chat template error: {exc}")

        # Format chat history for text prompt fallback
        history_str = ""
        if chat_history:
            for msg in chat_history:
                role = "Human" if hasattr(msg, "type") and msg.type == "human" else "AI"
                content = msg.content if hasattr(msg, "content") else str(msg)
                history_str += f"{role}: {content}\n"

        # Use the format_rag_prompt function from prompts module
        return format_rag_prompt(
            system_prompt=self.system_prompt,
            context=context,
            chat_history=history_str,
            question=question,
        )

    def invoke(self, inputs: LLMInputs) -> str:
        question = inputs.question
        chat_history = inputs.chat_history

        # Get context from retriever
        docs = self.retriever.invoke(question)
        context = self._format_docs(docs)

        # Format prompt
        prompt_input = self._format_prompt(question, chat_history, context)

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
        question = inputs.question
        chat_history = inputs.chat_history

        # Get context from retriever
        docs = self.retriever.invoke(question)
        context = self._format_docs(docs)

        # Format prompt
        prompt_input = self._format_prompt(question, chat_history, context)

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
