"""Base launcher interface for AI Interviewee."""

from abc import ABC, abstractmethod

from src.config import Config
from src.rag_engine import RAGEngine


class BaseLauncher(ABC):
    """Abstract base class for all launchers."""

    def __init__(self, config: Config, engine: RAGEngine):
        """Initialize the launcher.

        Args:
            config: Application configuration
            engine: RAG engine instance
        """
        self.config = config
        self.engine = engine

    @abstractmethod
    def launch(self) -> None:
        """Launch the application interface."""
        pass
