"""OpenAI API implementation."""

import logging
from collections.abc import Iterator
from typing import TYPE_CHECKING, Callable

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI

from src.llm.base import LLMInputs, LLMInterface

if TYPE_CHECKING:
    from src.config import Config

logger = logging.getLogger(__name__)


class OpenAILLM(LLMInterface):
    """OpenAI API implementation."""

    def __init__(self, system_prompt_fn: Callable[[str], str]):
        """Initialize OpenAI LLM.

        Args:
            system_prompt_fn: Function that takes user_name and returns system prompt

        Call initialize() before using invoke() or stream().
        """
        self.system_prompt_fn = system_prompt_fn

        # These will be set during initialize()
        self.config: Config
        self.llm: ChatOpenAI
        self.retriever: BaseRetriever
        self.system_prompt: str
        self.prompt_template: ChatPromptTemplate

    def initialize(self, config, retriever: BaseRetriever, user_name: str) -> None:
        """Initialize the LLM with config, retriever and user name.

        Args:
            config: Configuration instance
            retriever: Vector store retriever for RAG
            user_name: Name of the user/candidate
        """
        self.config = config

        if not self.config.openai_api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        self.retriever = retriever
        self.system_prompt = self.system_prompt_fn(user_name)

        # Get model settings
        self.settings = self.config.get_model_settings()

        # Get model name from config
        model_name = self.config.get_model_name()

        # Create LangChain ChatOpenAI instance
        self.llm = ChatOpenAI(
            model=model_name,
            temperature=self.settings.temperature,
            max_completion_tokens=self.settings.max_tokens,
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
        )

        logger.info(f"OpenAI LLM initialized with model: {model_name}")

    def _format_docs(self, docs):
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

    def invoke(self, inputs: LLMInputs) -> str:
        question = inputs.question
        chat_history = inputs.chat_history

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

    def stream(self, inputs: LLMInputs) -> Iterator[str]:
        question = inputs.question
        chat_history = inputs.chat_history

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
        return f"OpenAILLM(model={self.config.get_model_name()})"
