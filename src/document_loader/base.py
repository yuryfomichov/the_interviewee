"""Base interface for document loader implementations."""

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.vectorstores import VectorStoreRetriever


class DocumentLoaderInterface(ABC):
    """Abstract base class for document loader implementations."""

    @abstractmethod
    def initialize(self, force_rebuild: bool = False):
        """Initialize the document loading pipeline.

        Args:
            force_rebuild: Force rebuild of vector database

        Returns:
            Vector store instance
        """
        pass

    @abstractmethod
    def search(self, query: str, k: int | None = None) -> list[tuple]:
        """Search for relevant documents.

        Args:
            query: Search query
            k: Number of results to return

        Returns:
            List of (document, score) tuples
        """
        pass

    @abstractmethod
    def load_documents(self) -> list:
        """Load all documents from configured source.

        Returns:
            List of loaded documents
        """
        pass

    @abstractmethod
    def split_documents(self, documents: list) -> list:
        """Split documents into chunks.

        Args:
            documents: List of documents to split

        Returns:
            List of document chunks
        """
        pass

    @abstractmethod
    def get_retriever(self) -> "VectorStoreRetriever":
        """Get vector store retriever for use with LangChain.

        Returns:
            LangChain VectorStoreRetriever instance
        """
        pass

    @abstractmethod
    def __repr__(self) -> str:
        """String representation of the document loader."""
        pass
