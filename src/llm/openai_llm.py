"""OpenAI API implementation."""

import logging
from collections.abc import Generator
from typing import Any

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from src.config import get_config
from src.llm.base import LLMInterface
from src.prompts import get_system_prompt

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

    def get_system_prompt(self, user_name: str) -> str:
        """Get the system prompt for OpenAI models.

        Args:
            user_name: Name of the user/candidate

        Returns:
            System prompt string optimized for OpenAI models
        """
        return get_system_prompt(user_name)

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

    def create_rag_chain(
        self,
        retriever: Any,
        memory: Any,  # noqa: ARG002 - kept for API compatibility
        system_prompt: str,
    ) -> Any:
        """Create a RAG chain with memory for OpenAI.

        Args:
            retriever: Vector store retriever
            memory: Chat message history (not used - history managed externally)
            system_prompt: System prompt template

        Returns:
            Callable that generates responses with streaming support
        """
        # Create LangChain ChatOpenAI instance
        langchain_llm = ChatOpenAI(
            model=self.config.openai_model_name,
            temperature=self.config.openai_temperature,
            max_completion_tokens=self.config.openai_max_tokens,
            streaming=True,
        )

        # Format docs helper
        def format_docs(docs):
            if not docs:
                return "No specific career information retrieved."
            context_parts = []
            for i, doc in enumerate(docs, 1):
                # Handle both Document objects and strings
                if isinstance(doc, str):
                    context_parts.append(f"[Source {i}]\n{doc}\n")
                else:
                    source = (
                        doc.metadata.get("source", "Unknown")
                        if hasattr(doc, "metadata")
                        else "Unknown"
                    )
                    source_name = source.split("/")[-1] if "/" in source else source
                    page_content = doc.page_content if hasattr(doc, "page_content") else str(doc)
                    context_parts.append(f"[Source {i}: {source_name}]\n{page_content}\n")
            return "\n".join(context_parts)

        # Create a simple chain object that supports both streaming and non-streaming
        class OpenAIRAGChain:
            """Simple RAG chain for OpenAI that properly supports streaming."""

            def __init__(self, llm, retriever, system_prompt):
                self.llm = llm
                self.retriever = retriever
                self.system_prompt = system_prompt

                # Create prompt template
                self.prompt = ChatPromptTemplate.from_messages(
                    [
                        (
                            "system",
                            f"""{system_prompt}

Context from your career:
{{context}}""",
                        ),
                        MessagesPlaceholder(variable_name="chat_history"),
                        ("human", "{question}"),
                    ]
                )

            def invoke(self, inputs: dict, config: dict | None = None) -> Any:  # noqa: ARG002
                """Non-streaming invocation."""
                question = inputs.get("question", "")
                chat_history = inputs.get("chat_history", [])

                # Get context from retriever
                docs = self.retriever.invoke(question)
                context = format_docs(docs)

                # Format prompt
                messages = self.prompt.format_messages(
                    context=context,
                    chat_history=chat_history,
                    question=question,
                )

                # Generate response (non-streaming)
                response = self.llm.invoke(messages)
                return response.content

            def stream(self, inputs: dict, config: dict | None = None):  # noqa: ARG002
                """Streaming invocation."""
                question = inputs.get("question", "")
                chat_history = inputs.get("chat_history", [])

                # Get context from retriever
                docs = self.retriever.invoke(question)
                context = format_docs(docs)

                # Format prompt
                messages = self.prompt.format_messages(
                    context=context,
                    chat_history=chat_history,
                    question=question,
                )

                # Stream response
                for chunk in self.llm.stream(messages):
                    yield chunk.content

        return OpenAIRAGChain(langchain_llm, retriever, system_prompt)

    def __repr__(self) -> str:
        """String representation of the LLM."""
        return f"OpenAILLM(model={self.config.openai_model_name})"
