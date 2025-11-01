"""Factory for creating launcher instances."""

import logging

from src.config import Config
from src.launchers.base import BaseLauncher
from src.launchers.cli_launcher import CLILauncher
from src.launchers.gradio_launcher import GradioLauncher
from src.rag_engine import RAGEngine

logger = logging.getLogger(__name__)


def create_launcher(config: Config, engine: RAGEngine) -> BaseLauncher:
    """Create a launcher instance based on configuration.

    Args:
        config: Application configuration
        engine: RAG engine instance

    Returns:
        Launcher instance

    Raises:
        ValueError: If launcher type is not supported
    """
    launcher_type = config.launcher_type.lower()

    launchers = {
        "gradio": GradioLauncher,
        "cli": CLILauncher,
    }

    launcher_class = launchers.get(launcher_type)

    if launcher_class is None:
        available = ", ".join(launchers.keys())
        raise ValueError(
            f"Unsupported launcher type: {launcher_type}. "
            f"Available launchers: {available}"
        )

    logger.info(f"Creating {launcher_type} launcher")
    return launcher_class(config, engine)
