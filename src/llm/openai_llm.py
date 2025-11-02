"""OpenAI API implementation."""

import logging
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING, cast

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.retrievers import BaseRetriever
from langchain_openai import ChatOpenAI

from src.llm.base import LLMInputs, LLMInterface

if TYPE_CHECKING:
    from src.config import Config

from src.config import OpenAIModelSettings

logger = logging.getLogger(__name__)


class OpenAILLM(LLMInterface):
    """OpenAI API implementation."""

    llm: ChatOpenAI
    prompt_template: ChatPromptTemplate
    settings: OpenAIModelSettings

    def __init__(
        self,
        config: "Config",
        retriever: BaseRetriever,
        user_name: str,
        system_prompt_fn: Callable[[str], str],
    ):
        """Instantiate the OpenAI-backed LLM with all required dependencies."""
        super().__init__(
            config=config,
            retriever=retriever,
            user_name=user_name,
            system_prompt_fn=system_prompt_fn,
        )

        if not self.config.openai_api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")

        # Get model settings
        settings = self.config.get_model_settings()
        if not isinstance(settings, OpenAIModelSettings):
            raise TypeError("OpenAILLM requires OpenAI model settings.")
        self.settings = cast(OpenAIModelSettings, settings)

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

    def _serialize_messages(self, messages) -> str:
        """Serialize LangChain messages for logging."""
        serialized = []
        for msg in messages:
            role = getattr(msg, "type", getattr(msg, "role", ""))
            content = msg.content if hasattr(msg, "content") else str(msg)
            serialized.append(f"{role.upper()}: {content}")
        return "\n".join(serialized)

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
        self.last_prompt = self._serialize_messages(messages)

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
        self.last_prompt = self._serialize_messages(messages)

    def __repr__(self) -> str:
        """String representation of the LLM."""
        return f"OpenAILLM(model={self.config.get_model_name()})"
