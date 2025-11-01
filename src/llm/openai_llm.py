"""OpenAI API implementation."""

import logging
from collections.abc import Iterator
from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from src.config import get_config
from src.llm.base import LLMInterface
from src.prompts import get_system_prompt

logger = logging.getLogger(__name__)


class OpenAILLM(LLMInterface):
    """OpenAI API implementation."""

    def __init__(self, config=None, retriever: Any = None, user_name: str = ""):
        """Initialize OpenAI LLM.

        Args:
            config: Configuration instance (creates new if None)
            retriever: Vector store retriever for RAG
            user_name: Name of the user/candidate
        """
        self.config = config or get_config()

        if not self.config.openai_api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.retriever = retriever
        self.system_prompt = self.get_system_prompt(user_name) if user_name else ""

        # Create LangChain ChatOpenAI instance
        self.llm = ChatOpenAI(
            model=self.config.openai_model_name,
            temperature=self.config.openai_temperature,
            max_completion_tokens=self.config.openai_max_tokens,
            streaming=True,
        )

        # Create prompt template
        self.prompt_template = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""{self.system_prompt}

Context from your career:
{{context}}""",
                ),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", "{question}"),
            ]
        ) if self.system_prompt else None

        logger.info(f"Initialized OpenAI LLM with model: {self.config.openai_model_name}")

    def get_system_prompt(self, user_name: str) -> str:
        """Get the system prompt for OpenAI models.

        Args:
            user_name: Name of the user/candidate

        Returns:
            System prompt string optimized for OpenAI models
        """
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
                source = (
                    doc.metadata.get("source", "Unknown") if hasattr(doc, "metadata") else "Unknown"
                )
                source_name = source.split("/")[-1] if "/" in source else source
                page_content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                context_parts.append(f"[Source {i}: {source_name}]\n{page_content}\n")
        return "\n".join(context_parts)

    def invoke(self, inputs: dict) -> str:
        """Non-streaming invocation.

        Args:
            inputs: Dictionary containing question and chat_history

        Returns:
            Generated response as a string
        """
        if not self.llm or not self.prompt_template:
            raise RuntimeError("LLM not initialized properly.")

        question = inputs.get("question", "")
        chat_history = inputs.get("chat_history", [])

        # Get context from retriever
        docs = self.retriever.invoke(question)
        context = self._format_docs(docs)

        # Format prompt
        messages = self.prompt_template.format_messages(
            context=context,
            chat_history=chat_history,
            question=question,
        )

        # Generate response (non-streaming)
        response = self.llm.invoke(messages)
        content = response.content
        # Ensure we return a string
        if isinstance(content, str):
            return content
        return str(content)

    def stream(self, inputs: dict) -> Iterator[str]:
        """Streaming invocation.

        Args:
            inputs: Dictionary containing question and chat_history

        Yields:
            Response tokens/chunks as strings
        """
        if not self.llm or not self.prompt_template:
            raise RuntimeError("LLM not initialized properly.")

        question = inputs.get("question", "")
        chat_history = inputs.get("chat_history", [])

        # Get context from retriever
        docs = self.retriever.invoke(question)
        context = self._format_docs(docs)

        # Format prompt
        messages = self.prompt_template.format_messages(
            context=context,
            chat_history=chat_history,
            question=question,
        )

        # Stream response
        for chunk in self.llm.stream(messages):
            content = chunk.content
            # Ensure we yield strings
            if isinstance(content, str):
                yield content
            elif content:
                yield str(content)

    def __repr__(self) -> str:
        """String representation of the LLM."""
        return f"OpenAILLM(model={self.config.openai_model_name})"
