"""CLI launcher for AI Interviewee."""

import logging
import sys

from src.config import Config
from src.launchers.base import BaseLauncher
from src.rag_engine import RAGEngine

logger = logging.getLogger(__name__)


class CLILauncher(BaseLauncher):
    """Command-line interface launcher."""

    def __init__(self, config: Config, engine: RAGEngine):
        """Initialize the CLI launcher.

        Args:
            config: Application configuration
            engine: RAG engine instance
        """
        super().__init__(config, engine)

    def print_welcome(self) -> None:
        """Print welcome message."""
        print("\n" + "=" * 70)
        print(f"  {self.config.ui_title}")
        print("=" * 70)
        print(f"\n{self.config.ui_description}\n")
        print("Commands:")
        print("  - Type your interview question and press Enter")
        print("  - Type 'clear' to clear conversation history")
        print("  - Type 'quit' or 'exit' to exit the application")
        print("  - Press Ctrl+C to exit at any time")
        print("\n" + "-" * 70 + "\n")

    def print_system_info(self) -> None:
        """Print system information."""
        provider = self.config.model_provider
        model_name = self.config.get_model_name()

        if provider == "openai":
            device_info = "Cloud (OpenAI API)"
        else:
            settings = self.config.get_model_settings()
            device_info = settings.device if hasattr(settings, "device") else "N/A"

        print(f"Model Provider: {provider}")
        print(f"Model: {model_name}")
        print(f"Device: {device_info}")
        print(f"Embedding Model: {self.config.embedding_model}")
        print()

    def chat(self, message: str) -> None:
        """Process chat message and display response.

        Args:
            message: User message
        """
        if not message or not message.strip():
            return

        if not self.engine:
            print("\nError: RAG engine not initialized\n")
            return

        try:
            print(f"\n{self.config.user_name}: ", end="", flush=True)

            # Stream the response
            print()
            for token in self.engine.generate_response(message, stream=True):
                print(token, end="", flush=True)
            print("\n")

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            print(
                "\nI apologize, but I encountered an error. "
                "Please try again or rephrase your question.\n"
            )

    def clear_conversation(self) -> None:
        """Clear conversation history."""
        if self.engine:
            self.engine.clear_history()
        print("\nConversation history cleared.\n")
        logger.info("Conversation cleared")

    def launch(self) -> None:
        """Launch the CLI application."""
        self.print_welcome()

        # Show system info in verbose mode
        if logger.getEffectiveLevel() <= logging.INFO:
            self.print_system_info()

        try:
            while True:
                try:
                    # Get user input
                    user_input = input("You: ").strip()

                    # Handle empty input
                    if not user_input:
                        continue

                    # Handle commands
                    if user_input.lower() in ["quit", "exit"]:
                        print("\nGoodbye!\n")
                        break
                    elif user_input.lower() == "clear":
                        self.clear_conversation()
                        continue
                    elif user_input.lower() == "help":
                        self.print_welcome()
                        continue

                    # Process the message
                    self.chat(user_input)

                except EOFError:
                    # Handle Ctrl+D
                    print("\n\nGoodbye!\n")
                    break

        except KeyboardInterrupt:
            # Handle Ctrl+C
            print("\n\nGoodbye!\n")
            sys.exit(0)
