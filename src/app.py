"""Main entry point for AI Interviewee application."""

import logging
import sys

from src.config import get_config
from src.launchers import create_launcher
from src.rag_engine import create_rag_engine

logger = logging.getLogger(__name__)


def main():
    """Main entry point for the application."""
    try:
        # Load configuration
        config = get_config()
        logger.info(f"Loaded configuration: {config}")
        logger.info(f"Launcher type: {config.launcher_type}")

        # Initialize RAG engine
        logger.info("Initializing RAG engine...")
        engine = create_rag_engine(config)
        logger.info("RAG engine initialized successfully")

        # Create and launch the appropriate launcher
        launcher = create_launcher(config, engine)
        launcher.launch()

    except Exception as e:
        logger.error(f"Failed to start application: {e}", exc_info=True)
        print(f"\nError starting application: {e}")
        print("\nPlease check:")
        print("1. Configuration file (config.yaml) exists")
        print("2. Career data directory exists and contains markdown files")
        print("3. All dependencies are installed")
        print("4. Environment variables are set correctly (if using .env)")
        print("5. LAUNCHER_TYPE is set to 'gradio' or 'cli' in .env")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
