"""Configuration management for AI Interviewee."""

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


class MLXModelSettings(BaseModel):
    """Settings for MLX-based models (Qwen, Llama)."""

    device: str = Field(default="mps", description="Device to run the model on (mps, cuda, cpu)")
    max_new_tokens: int = Field(default=512, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")


class OpenAIModelSettings(BaseModel):
    """Settings for OpenAI API models."""

    max_tokens: int = Field(default=1024, description="Maximum number of tokens to generate")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Sampling temperature")


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
        self._load_model_settings()

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

    def _load_model_settings(self) -> None:
        """Load and validate model settings using Pydantic."""
        # Mapping of provider to settings model
        provider_settings_models = {
            "qwen": MLXModelSettings,
            "llama": MLXModelSettings,
            "openai": OpenAIModelSettings,
        }

        # Load settings for each provider
        self.model_settings: dict[str, MLXModelSettings | OpenAIModelSettings] = {}
        for provider, settings_model in provider_settings_models.items():
            settings_dict = self.get(f"model.{provider}.settings", {})
            try:
                self.model_settings[provider] = settings_model(**settings_dict)
                logger.info(f"Loaded {provider} settings: {self.model_settings[provider]}")
            except Exception as e:
                logger.warning(f"Failed to load {provider} settings: {e}. Using defaults.")
                self.model_settings[provider] = settings_model()

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
        """Get the LLM provider (qwen, llama, openai)."""
        return os.getenv("MODEL_PROVIDER", self.get("model.provider", "qwen"))

    def get_model_name(self) -> str:
        """Get the model name for the current provider.

        Returns:
            Model name for the current provider
        """
        return os.getenv("MODEL_NAME", self.get(f"model.{self.model_provider}.model_name", ""))

    def get_model_settings(self) -> MLXModelSettings | OpenAIModelSettings:
        """Get Pydantic settings model for a specific provider.


        Returns:
            Pydantic settings model for the provider
        """
        return self.model_settings.get(self.model_provider, MLXModelSettings())

    @property
    def openai_api_key(self) -> str | None:
        """Get OpenAI API key from environment."""
        return os.getenv("OPENAI_API_KEY")

    @property
    def huggingface_token(self) -> str | None:
        """Get HuggingFace token from environment."""
        return os.getenv("HUGGINGFACE_TOKEN")

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

    # Launcher configuration
    @property
    def launcher_type(self) -> str:
        """Get launcher type (gradio, cli)."""
        return os.getenv("LAUNCHER_TYPE", "gradio").lower()

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
        settings = self.get_model_settings()
        if isinstance(settings, MLXModelSettings):
            return f"Config(provider={self.model_provider}, device={settings.device})"
        return f"Config(provider={self.model_provider})"


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
