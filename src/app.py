"""Gradio web interface for AI Interviewee."""

import logging

import gradio as gr
from gradio.themes import Soft

from src.config import get_config
from src.rag_engine import RAGEngine

logger = logging.getLogger(__name__)


class InterviewApp:
    """Gradio application for interview assistant."""

    # Example interview questions
    EXAMPLE_QUESTIONS = [
        "Tell me about yourself",
        "What's your most significant achievement?",
        "Describe a challenging project you worked on",
        "Why are you interested in this role?",
        "Tell me about a time you overcame a difficult obstacle",
        "What's your leadership style?",
    ]

    def __init__(self):
        """Initialize the application."""
        self.config = get_config()
        self.engine = None
        self._initialize_engine()

    def _initialize_engine(self) -> None:
        """Initialize the RAG engine."""
        logger.info("Initializing RAG engine...")
        try:
            self.engine = RAGEngine(self.config)
            logger.info("RAG engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize RAG engine: {e}")
            raise

    def chat(self, message: str, history: list):
        """Process chat message and stream response.

        Args:
            message: User message
            history: Chat history

        Yields:
            Accumulated response for streaming display
        """
        if not message or not message.strip():
            return

        if not self.engine:
            yield "Error: RAG engine not initialized"
            return

        try:
            # Stream the response, accumulating tokens
            full_response = ""
            for token in self.engine.generate_response(message, stream=True):
                full_response += token
                yield full_response

        except Exception as e:
            logger.error(f"Error processing message: {e}")
            yield (
                "I apologize, but I encountered an error. "
                "Please try again or rephrase your question."
            )

    def clear_conversation(self) -> None:
        """Clear conversation history."""
        if self.engine:
            self.engine.clear_history()
        logger.info("Conversation cleared")

    def create_interface(self) -> gr.Blocks:
        """Create Gradio interface.

        Returns:
            Gradio Blocks interface
        """
        with gr.Blocks(
            title=self.config.ui_title,
            theme=Soft(),
        ) as demo:
            # Header
            gr.Markdown(
                f"""
                # {self.config.ui_title}
                {self.config.ui_description}
                """
            )

            # Main chat interface
            chatbot = gr.ChatInterface(
                fn=self.chat,
                type="messages",
                chatbot=gr.Chatbot(
                    height=500,
                    show_copy_button=True,
                ),
                textbox=gr.Textbox(
                    placeholder="Ask me an interview question...",
                    container=False,
                    scale=7,
                    submit_btn="Send",
                ),
            )

            # Example questions section
            if self.config.show_examples:
                with gr.Accordion("Example Questions", open=False):
                    gr.Markdown("Click on any question to try it:")

                    for question in self.EXAMPLE_QUESTIONS:
                        btn = gr.Button(question, size="sm")
                        btn.click(
                            fn=lambda q=question: q,
                            outputs=chatbot.textbox,
                        )

            # Information section
            with gr.Accordion("About", open=False):
                gr.Markdown(
                    f"""
                    ### System Information
                    - **Model Provider**: {self.config.model_provider}
                    - **Model**: {self.config.local_model_name if self.config.model_provider == "local" else self.config.openai_model_name}
                    - **Device**: {self.config.local_model_device}
                    - **Embedding Model**: {self.config.embedding_model}

                    ### How it works
                    This AI assistant uses Retrieval-Augmented Generation (RAG) to answer interview questions
                    based on career documents. It retrieves relevant information from your professional background
                    and generates natural, contextual responses.

                    ### Tips for best results
                    - Ask specific questions about experience, skills, and projects
                    - Use common interview question formats
                    - The assistant will politely decline non-professional questions
                    """
                )

        return demo

    def launch(self, **kwargs) -> None:
        """Launch the Gradio application.

        Args:
            **kwargs: Additional arguments to pass to gr.Blocks.launch()
        """
        demo = self.create_interface()

        # Set default launch parameters
        launch_params = {
            "server_name": self.config.gradio_server_name,
            "server_port": self.config.gradio_server_port,
            "share": self.config.gradio_share,
            "show_error": True,
        }

        # Override with any provided kwargs
        launch_params.update(kwargs)

        logger.info(
            f"Launching Gradio app at http://{launch_params['server_name']}:{launch_params['server_port']}"
        )

        demo.launch(**launch_params)


def main():
    """Main entry point for the application."""
    try:
        app = InterviewApp()
        app.launch()
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        print(f"\n❌ Error starting application: {e}")
        print("\nPlease check:")
        print("1. Configuration file (config.yaml) exists")
        print("2. Career data directory exists and contains markdown files")
        print("3. All dependencies are installed")
        print("4. Environment variables are set correctly (if using .env)")
        return 1

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
