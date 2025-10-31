"""RAG (Retrieval-Augmented Generation) engine for AI Interviewee."""

import logging
from collections.abc import Iterator

from langchain_community.chat_message_histories import ChatMessageHistory

from src.config import Config, get_config
from src.document_loader import DocumentLoaderInterface, create_document_loader
from src.llm import LLMInterface, create_llm
from src.prompts import OUT_OF_SCOPE_RESPONSE

logger = logging.getLogger(__name__)


class RAGEngine:
    """Retrieval-Augmented Generation engine for interview responses."""

    def __init__(
        self,
        llm: LLMInterface,
        document_loader: DocumentLoaderInterface,
        config: Config | None = None,
    ):
        """Initialize RAG engine with injected dependencies.

        Args:
            llm: LLM interface instance
            document_loader: Document loader interface instance
            config: Configuration instance (creates new if None)
        """
        self.config = config or get_config()
        self.llm = llm
        self.document_loader = document_loader

        # Get system prompt from LLM (each LLM can customize its prompt)
        self.SYSTEM_PROMPT = self.llm.get_system_prompt(self.config.user_name)

        # Create conversational memory
        self.memory = ChatMessageHistory()

        # Create retriever from document loader
        self.retriever = self.document_loader.get_retriever()

        # Create RAG chain from LLM
        self.chain = self.llm.create_rag_chain(
            retriever=self.retriever,
            memory=self.memory,
            system_prompt=self.SYSTEM_PROMPT,
        )

        logger.info(f"RAG engine initialized with {self.llm} and {self.document_loader}")

    @classmethod
    def create_default(cls, config: Config | None = None) -> "RAGEngine":
        """Factory method to create RAG engine with default dependencies.

        Args:
            config: Configuration instance (creates new if None)

        Returns:
            RAGEngine instance with default dependencies
        """
        config = config or get_config()
        logger.info("Creating RAG engine with default dependencies")

        # Create dependencies
        llm = create_llm(config)
        document_loader = create_document_loader(config)
        document_loader.initialize()

        return cls(llm=llm, document_loader=document_loader, config=config)

    def _is_out_of_scope(self, question: str) -> bool:
        """Check if question is out of scope (basic heuristic).

        Args:
            question: User question

        Returns:
            True if question appears out of scope
        """
        # Keywords that might indicate out-of-scope questions
        out_of_scope_keywords = [
            "personal life",
            "family",
            "religion",
            "political",
            "politics",
            "confidential",
            "other candidates",
            "salary history",
            "age",
            "marital status",
        ]

        question_lower = question.lower()

        for keyword in out_of_scope_keywords:
            if keyword in question_lower:
                logger.info(f"Question flagged as potentially out-of-scope: {keyword}")
                return True

        return False

    def generate_response(
        self, question: str, use_history: bool = True, stream: bool = False
    ) -> Iterator[str]:
        """Generate response to interview question.

        Args:
            question: User question
            use_history: Whether to consider conversation history
            stream: Whether to stream the response

        Yields:
            Response tokens
        """
        logger.info(f"Generating response for: {question[:100]}...")

        # Check if question is out of scope
        if self._is_out_of_scope(question):
            yield OUT_OF_SCOPE_RESPONSE
            return

        try:
            # Get chat history if needed
            chat_history = self.memory.messages if use_history else []

            # Prepare inputs
            inputs = {"question": question, "chat_history": chat_history}

            # Collect full response for history
            full_response = ""

            if stream:
                # Streaming - manually manage history
                for chunk in self.chain.stream(inputs):
                    # chunk is a string token
                    token = chunk if isinstance(chunk, str) else str(chunk)
                    full_response += token
                    yield token
                logger.info("Streaming response generated successfully")
            else:
                # Non-streaming - manually manage history
                result = self.chain.invoke(inputs)
                response = result if isinstance(result, str) else str(result)
                full_response = response
                yield response
                logger.info("Response generated successfully")

            # Manually add to history
            from langchain_core.messages import AIMessage, HumanMessage

            self.memory.add_message(HumanMessage(content=question))
            self.memory.add_message(AIMessage(content=full_response))

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            error_msg = (
                "I apologize, but I encountered an error. "
                "Please try again or rephrase your question."
            )
            yield error_msg

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.memory.clear()
        logger.info("Conversation history cleared")

    def get_history(self) -> list:
        """Get conversation history.

        Returns:
            List of conversation messages
        """
        return self.memory.messages

    def __repr__(self) -> str:
        """String representation of RAG engine."""
        return f"RAGEngine(llm={self.llm}, document_loader={self.document_loader})"
