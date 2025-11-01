"""Apple Silicon optimized LLM using MLX."""

import logging
from collections.abc import Iterator
from typing import Any

from mlx_lm.generate import stream_generate
from mlx_lm.utils import load as mlx_load

from src.config import get_config
from src.llm.base import LLMInterface
from src.prompts import format_rag_prompt, get_system_prompt, get_system_prompt_for_qwen

logger = logging.getLogger(__name__)


class MLXLocalLLM(LLMInterface):
    """Apple Silicon optimized LLM using MLX with LangChain integration."""

    def __init__(self, config=None, retriever: Any = None, user_name: str = ""):
        """Initialize MLX LLM using mlx_lm directly.

        Args:
            config: Configuration instance (creates new if None)
            retriever: Vector store retriever for RAG
            user_name: Name of the user/candidate
        """
        self.config = config or get_config()

        model_name = self.config.local_model_name
        logger.info(f"Loading model with MLX: {model_name}")
        logger.info("Device: Apple Silicon (Metal)")

        # Prepare authentication token for gated models
        token = self.config.huggingface_token
        if token:
            logger.info("Using HuggingFace authentication token")
            import os

            os.environ["HF_TOKEN"] = token

        try:
            # Load model and tokenizer using mlx_lm directly (single load for both streaming and non-streaming)
            logger.info("Loading model with mlx_lm...")
            model_artifacts = mlx_load(model_name)
            # mlx_load can return 2 or 3 items (model, tokenizer, [config])
            self.model = model_artifacts[0]
            self.tokenizer = model_artifacts[1]

            # Don't load a second copy via MLXPipeline - use mlx_lm directly for everything
            self.mlx_pipeline = None

            # Set retriever and system prompt
            self.retriever = retriever
            self.system_prompt = self.get_system_prompt(user_name) if user_name else ""

            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def get_system_prompt(self, user_name: str) -> str:
        """Get the system prompt for MLX local models.

        Automatically detects Qwen models and uses optimized prompt.

        Args:
            user_name: Name of the user/candidate

        Returns:
            System prompt string optimized for the specific model
        """
        model_name = self.config.local_model_name.lower()

        # Use Qwen-specific prompt for Qwen models
        if "qwen" in model_name:
            logger.info("Using Qwen-optimized system prompt")
            return get_system_prompt_for_qwen(user_name)

        # Default prompt for other models
        return get_system_prompt(user_name)

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
        # Format chat history
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

    def invoke(self, inputs: dict) -> str:
        """Non-streaming invocation.

        Args:
            inputs: Dictionary containing question and chat_history

        Returns:
            Generated response as a string
        """
        question = inputs.get("question", "")
        chat_history = inputs.get("chat_history", [])

        # Get context from retriever
        docs = self.retriever.invoke(question) if self.retriever else []
        context = self._format_docs(docs)

        # Format prompt
        full_prompt = self._format_prompt(question, chat_history, context)

        # Generate response (non-streaming) using mlx_lm directly
        from mlx_lm.generate import generate as mlx_generate

        response = mlx_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=full_prompt,
            max_tokens=self.config.local_model_max_tokens,
            verbose=False,
        )
        return response

    def stream(self, inputs: dict) -> Iterator[str]:
        """Streaming invocation.

        Args:
            inputs: Dictionary containing question and chat_history

        Yields:
            Response tokens/chunks as strings
        """
        question = inputs.get("question", "")
        chat_history = inputs.get("chat_history", [])

        # Get context from retriever
        docs = self.retriever.invoke(question) if self.retriever else []
        context = self._format_docs(docs)

        # Format prompt
        full_prompt = self._format_prompt(question, chat_history, context)

        # Stream response using mlx_lm directly (MLXPipeline streaming is broken)
        for generation_response in stream_generate(
            model=self.model,
            tokenizer=self.tokenizer,
            prompt=full_prompt,
            max_tokens=self.config.local_model_max_tokens,
        ):
            # stream_generate yields GenerationResponse objects with a text attribute
            yield generation_response.text

    def __repr__(self) -> str:
        """String representation of the LLM."""
        return f"MLXLocalLLM(model={self.config.local_model_name}, device=Apple Silicon)"
