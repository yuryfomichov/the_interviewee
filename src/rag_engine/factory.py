"""Factory function for creating RAG engine."""

import logging

from src.config import Config, get_config
from src.document_loader import create_document_loader
from src.llm import create_llm
from src.rag_engine.engine import RAGEngine

logger = logging.getLogger(__name__)


def create_rag_engine(config: Config | None = None) -> RAGEngine:
    """Create RAG engine with default dependencies.

    Args:
        config: Configuration instance (creates new if None)

    Returns:
        RAGEngine instance with all dependencies initialized
    """
    config = config or get_config()
    logger.info("Creating RAG engine with default dependencies")

    # Create and initialize document loader
    document_loader = create_document_loader(config)
    document_loader.initialize()

    retriever = document_loader.get_retriever()

    # Create fully initialized LLM
    llm = create_llm(
        retriever=retriever,
        config=config,
        user_name=config.user_name,
    )

    # Pass LLM directly to RAGEngine
    return RAGEngine(llm=llm, config=config)
