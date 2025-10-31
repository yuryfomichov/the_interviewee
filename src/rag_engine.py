"""RAG (Retrieval-Augmented Generation) engine for AI Interviewee."""

import logging

from src.config import get_config
from src.document_loader import DocumentLoader
from src.llm import create_llm

logger = logging.getLogger(__name__)


class RAGEngine:
    """Retrieval-Augmented Generation engine for interview responses."""

    OUT_OF_SCOPE_TEMPLATE = """I appreciate the question, but that's outside the scope of my professional background that I can discuss. I'd be happy to talk more about my relevant experience, skills, and projects. Is there anything specific about my career you'd like to know more about?"""

    def __init__(self, config=None):
        """Initialize RAG engine.

        Args:
            config: Configuration instance (creates new if None)
        """
        self.config = config or get_config()
        self.conversation_history: list = []

        # System prompt template - uses user_name from config
        self.SYSTEM_PROMPT_TEMPLATE = f"""You are {self.config.user_name}, an experienced professional in an interview. Answer the question directly and naturally, as if speaking to an interviewer.

CRITICAL RULES:
- Answer ONLY the specific question asked
- Do NOT generate follow-up questions or continue the conversation
- Do NOT add notes, explanations, or meta-commentary after your answer
- Do NOT include phrases like "Note:", "Final output:", "End of response", "Question:", etc.
- Do NOT explain your answer structure
- Keep answers concise and focused (2-3 paragraphs maximum)
- Use the STAR method (Situation, Task, Action, Result) for behavioral questions
- Speak naturally in first person, as if in a conversation
- STOP after answering the question

Context from your career:
{{context}}

Question: {{question}}

Answer:"""

        # Initialize document loader and vector store
        logger.info("Initializing RAG engine")
        self.document_loader = DocumentLoader(self.config)
        self.document_loader.initialize()

        # Initialize LLM
        self.llm = create_llm(self.config)

        logger.info(f"RAG engine initialized with {self.llm}")

    def _format_context(self, retrieved_docs: list) -> str:
        """Format retrieved documents into context string.

        Args:
            retrieved_docs: List of (document, score) tuples

        Returns:
            Formatted context string
        """
        if not retrieved_docs:
            return "No specific career information retrieved."

        context_parts = []
        for i, (doc, _) in enumerate(retrieved_docs, 1):
            # Extract source file name
            source = doc.metadata.get("source", "Unknown")
            source_name = source.split("/")[-1] if "/" in source else source

            context_parts.append(f"[Source {i}: {source_name}]\n{doc.page_content}\n")

        return "\n".join(context_parts)

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

    def retrieve_context(self, question: str) -> tuple[str, list]:
        """Retrieve relevant context for a question.

        Args:
            question: User question

        Returns:
            Tuple of (formatted_context, list of (doc, score) tuples)
        """
        logger.info(f"Retrieving context for question: {question[:100]}...")

        # Search for relevant documents
        retrieved_docs = self.document_loader.search(question, k=self.config.top_k)

        if not retrieved_docs:
            logger.warning("No relevant documents found for question")

        logger.info(f"Retrieved {len(retrieved_docs)} relevant document chunks")

        # Format context
        context = self._format_context(retrieved_docs)

        return context, retrieved_docs

    def generate_response(self, question: str, use_history: bool = True, stream: bool = False):
        """Generate response to interview question.

        Args:
            question: User question
            use_history: Whether to consider conversation history
            stream: Whether to stream the response

        Returns:
            Dictionary containing response info, or generator if streaming
        """
        logger.info(f"Generating response for: {question[:100]}...")

        result = {
            "response": "",
            "context": "",
            "sources": [],
            "out_of_scope": False,
        }

        # Check if question is out of scope
        if self._is_out_of_scope(question):
            result["response"] = self.OUT_OF_SCOPE_TEMPLATE
            result["out_of_scope"] = True
            if stream:
                yield result["response"]
                return
            return result

        # Retrieve context
        context, retrieved_docs = self.retrieve_context(question)
        result["context"] = context
        result["sources"] = [doc.metadata.get("source", "Unknown") for doc, _ in retrieved_docs]

        # Build prompt with context
        prompt = self.SYSTEM_PROMPT_TEMPLATE.format(context=context, question=question)

        # Generate response
        try:
            if stream:
                # Stream the response
                full_response = ""
                for token in self.llm.generate(prompt, stream=True):
                    full_response += token
                    yield token

                # Update conversation history after streaming completes
                if use_history and self.config.enable_history:
                    self.conversation_history.append(
                        {"question": question, "answer": full_response}
                    )
                logger.info("Streaming response generated successfully")
            else:
                # Non-streaming response - consume the generator
                response = "".join(self.llm.generate(prompt, stream=False))
                result["response"] = response

                # Update conversation history if enabled
                if use_history and self.config.enable_history:
                    self.conversation_history.append({"question": question, "answer": response})

                logger.info("Response generated successfully")
                return result

        except Exception as e:
            logger.error(f"Error generating response: {e}")

        return result

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
        logger.info("Conversation history cleared")

    def get_history(self) -> list:
        """Get conversation history.

        Returns:
            List of conversation turns
        """
        return self.conversation_history.copy()

    def __repr__(self) -> str:
        """String representation of RAG engine."""
        return f"RAGEngine(llm={self.llm}, docs_loaded={self.document_loader.vector_store is not None})"
