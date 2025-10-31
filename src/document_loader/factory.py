"""Factory function for creating document loader instances."""

import logging

from src.config import Config, get_config
from src.document_loader.base import DocumentLoaderInterface

logger = logging.getLogger(__name__)


def create_document_loader(config: Config | None = None) -> DocumentLoaderInterface:
    """Factory function to create appropriate document loader based on configuration.

    Currently supports HuggingFace embeddings. Future implementations could include:
    - OpenAI embeddings
    - Custom embeddings
    - Different vector stores (Pinecone, Weaviate, etc.)

    Args:
        config: Configuration instance (creates new if None)

    Returns:
        DocumentLoaderInterface instance
    """
    config = config or get_config()

    # For now, we only have HuggingFace implementation
    # In the future, this could be configurable via config.embedding_provider
    logger.info("Creating HuggingFace document loader")
    from src.document_loader.huggingface_loader import HuggingFaceDocumentLoader

    return HuggingFaceDocumentLoader(config)
