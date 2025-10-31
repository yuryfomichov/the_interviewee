from src.config import Config


def test_config_loading():
    """Test that configuration loads successfully."""
    config = Config("config.yaml")
    assert config is not None
    assert config.model_provider in ["local", "openai"]


def test_config_properties():
    """Test configuration properties."""
    config = Config("config.yaml")

    # Model properties
    assert isinstance(config.chunk_size, int)
    assert config.chunk_size > 0

    # RAG properties
    assert isinstance(config.top_k, int)
    assert config.top_k > 0

    # Path properties
    assert config.career_data_path is not None
    assert config.vector_db_path is not None
