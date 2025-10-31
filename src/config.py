"""Configuration management for AI Interviewee."""

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class Config:
    """Application configuration manager."""

    def __init__(self, config_path: str = "config.yaml"):
        """Initialize configuration from YAML file and environment variables.

        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self._config: dict[str, Any] = {}
        self._load_config()
        self._setup_logging()

    def _load_config(self) -> None:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path) as f:
            self._config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {self.config_path}")

    def _setup_logging(self) -> None:
        """Set up logging configuration."""
        log_level = os.getenv("LOG_LEVEL", "INFO").upper()
        logging.basicConfig(
            level=getattr(logging, log_level),
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path to configuration value (e.g., 'model.local.device')
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key_path.split(".")
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    # Model configuration properties
    @property
    def model_provider(self) -> str:
        """Get the LLM provider (local, openai)."""
        return os.getenv("MODEL_PROVIDER", self.get("model.provider", "local"))

    @property
    def local_model_name(self) -> str:
        """Get the local model name."""
        return os.getenv("LOCAL_MODEL_NAME", self.get("model.local.model_name"))

    @property
    def local_model_device(self) -> str:
        """Get the device for local model (mps, cuda, cpu)."""
        return os.getenv("DEVICE", self.get("model.local.device", "cpu"))

    @property
    def local_model_max_tokens(self) -> int:
        """Get max new tokens for local model."""
        return self.get("model.local.max_new_tokens", 512)

    @property
    def local_model_temperature(self) -> float:
        """Get temperature for local model."""
        return self.get("model.local.temperature", 0.7)

    @property
    def openai_api_key(self) -> str | None:
        """Get OpenAI API key from environment."""
        return os.getenv("OPENAI_API_KEY")

    @property
    def huggingface_token(self) -> str | None:
        """Get HuggingFace token from environment."""
        return os.getenv("HUGGINGFACE_TOKEN")

    @property
    def openai_model_name(self) -> str:
        """Get OpenAI model name."""
        return self.get("model.openai.model_name", "gpt-4")

    @property
    def openai_max_tokens(self) -> int:
        """Get max tokens for OpenAI model."""
        return self.get("model.openai.max_tokens", 1024)

    @property
    def openai_temperature(self) -> float:
        """Get temperature for OpenAI model."""
        return self.get("model.openai.temperature", 0.7)

    # RAG configuration properties
    @property
    def chunk_size(self) -> int:
        """Get document chunk size."""
        return self.get("rag.chunk_size", 800)

    @property
    def chunk_overlap(self) -> int:
        """Get chunk overlap size."""
        return self.get("rag.chunk_overlap", 200)

    @property
    def top_k(self) -> int:
        """Get number of top documents to retrieve."""
        return self.get("rag.top_k", 5)

    @property
    def relevance_threshold(self) -> float:
        """Get relevance score threshold."""
        return self.get("rag.relevance_threshold", 0.5)

    @property
    def embedding_model(self) -> str:
        """Get embedding model name."""
        return self.get("rag.embedding_model", "BAAI/bge-base-en-v1.5")

    # Data configuration properties
    @property
    def career_data_path(self) -> Path:
        """Get career data directory path."""
        path = os.getenv("CAREER_DATA_PATH", self.get("data.career_data_path", "./career_data"))
        return Path(path)

    @property
    def vector_db_path(self) -> Path:
        """Get vector database path."""
        path = os.getenv("VECTOR_DB_PATH", self.get("data.vector_db_path", "./vector_db"))
        return Path(path)

    @property
    def models_cache_path(self) -> Path:
        """Get models cache directory path."""
        path = os.getenv("MODELS_CACHE_PATH", self.get("data.models_cache_path", "./models"))
        return Path(path)

    @property
    def rebuild_index(self) -> bool:
        """Check if vector index should be rebuilt."""
        return self.get("data.rebuild_index", False)

    # User configuration properties
    @property
    def user_name(self) -> str:
        """Get user name from environment variable."""
        return os.getenv("USER_NAME", "User")

    # UI configuration properties
    @property
    def ui_title(self) -> str:
        """Get UI title."""
        default_title = (
            f"{self.user_name} - Interviewee" if self.user_name != "User" else "AI Interviewee"
        )
        return self.get("ui.title", default_title)

    @property
    def ui_description(self) -> str:
        """Get UI description."""
        return self.get(
            "ui.description", "Ask me questions about my professional background and experience."
        )

    @property
    def show_examples(self) -> bool:
        """Check if example questions should be shown."""
        return self.get("ui.show_examples", True)

    @property
    def enable_history(self) -> bool:
        """Check if conversation history should be enabled."""
        return self.get("ui.enable_history", True)

    # Gradio server configuration
    @property
    def gradio_server_name(self) -> str:
        """Get Gradio server name/host."""
        return os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")

    @property
    def gradio_server_port(self) -> int:
        """Get Gradio server port."""
        return int(os.getenv("GRADIO_SERVER_PORT", "7860"))

    @property
    def gradio_share(self) -> bool:
        """Check if Gradio should create a public link."""
        return os.getenv("GRADIO_SHARE", "false").lower() == "true"

    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"Config(provider={self.model_provider}, device={self.local_model_device})"


# Global configuration instance
_config: Config | None = None


def get_config(config_path: str = "config.yaml") -> Config:
    """Get or create global configuration instance.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration instance
    """
    global _config
    if _config is None:
        _config = Config(config_path)
    return _config
