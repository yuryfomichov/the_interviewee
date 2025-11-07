"""Pytest fixtures for prompt optimizer pipeline tests."""

import tempfile
from pathlib import Path

import pytest
from agents import Runner

from prompt_optimizer.config import LLMConfig, OptimizerConfig, TestDistribution
from prompt_optimizer.schemas import TaskSpec
from prompt_optimizer.storage import Database
from prompt_optimizer.tests.helpers import DummyConnector
from prompt_optimizer.tests.helpers.fake_agents import fake_runner_run


@pytest.fixture
def temp_db_path(tmp_path):
    """Provide a temporary database path for testing."""
    db_dir = tmp_path / "test_storage"
    db_dir.mkdir(exist_ok=True)
    return db_dir / "test_optimizer.db"


@pytest.fixture
def test_database(temp_db_path):
    """
    Provide a real Database instance with temporary storage.

    Uses real SQLite database (not in-memory) so we can inspect state between stages.
    Database is automatically cleaned up after test completes.
    """
    db = Database(temp_db_path)
    yield db
    # Cleanup happens automatically when tmp_path is removed


@pytest.fixture
def dummy_connector():
    """
    Provide a DummyConnector for fast, deterministic model responses.

    This replaces the real model connector (e.g., OpenAI) with a fake
    that generates responses instantly without API calls.
    """
    return DummyConnector(seed=42)


@pytest.fixture
def mock_agents(monkeypatch):
    """
    Mock the agents.Runner.run method to return fake responses.

    This prevents real LLM API calls while allowing the pipeline to run.
    The fake responses are deterministic and realistic.
    """
    # Patch Runner.run with our fake implementation
    monkeypatch.setattr(Runner, "run", fake_runner_run)


@pytest.fixture
def sample_task_spec():
    """Provide a sample task specification for testing."""
    return TaskSpec(
        task_description="Provide helpful and accurate answers to user questions",
        behavioral_specs=(
            "Be concise but thorough. Use clear language. "
            "Acknowledge uncertainty when appropriate. "
            "Stay within scope of the question."
        ),
        validation_rules=[
            "Responses must be relevant to the question",
            "Must not provide harmful or dangerous information",
            "Should maintain a professional and helpful tone",
            "Should acknowledge limitations when uncertain",
        ],
        current_prompt=None,
    )


@pytest.fixture
def sample_task_spec_with_original_prompt(sample_task_spec):
    """Provide a task spec with an original prompt for comparison testing."""
    spec = sample_task_spec.model_copy()
    spec.current_prompt = """You are a helpful AI assistant.

Your role is to answer user questions accurately and concisely.

Guidelines:
- Be clear and direct
- Admit when you don't know something
- Stay on topic
- Be professional and respectful

Always prioritize accuracy over speed."""
    return spec


@pytest.fixture
def minimal_config(sample_task_spec, tmp_path):
    """
    Provide minimal configuration for fast smoke tests.

    Uses small numbers to test pipeline logic quickly:
    - 3 initial prompts
    - 2 quick tests
    - 3 rigorous tests
    - Top 2 advance to rigorous
    - Top 2 go to refinement
    - Max 2 refinement iterations
    """
    return OptimizerConfig(
        # Stage 1: Quick filter
        num_initial_prompts=3,
        quick_test_distribution=TestDistribution(
            core=1, edge=1, boundary=0, adversarial=0, consistency=0, format=0
        ),
        top_k_advance=2,
        # Stage 2: Rigorous testing
        rigorous_test_distribution=TestDistribution(
            core=2, edge=1, boundary=0, adversarial=0, consistency=0, format=0
        ),
        top_m_refine=2,
        # Stage 3: Refinement
        max_iterations_per_track=2,
        convergence_threshold=0.02,
        early_stopping_patience=1,
        # LLM configs (just metadata - actual calls are mocked)
        generator_llm=LLMConfig(model="gpt-4o", temperature=0.8),
        test_designer_llm=LLMConfig(model="gpt-4o", temperature=0.7),
        evaluator_llm=LLMConfig(model="gpt-4o", temperature=0.3),
        refiner_llm=LLMConfig(model="gpt-4o", temperature=0.7),
        # Execution
        parallel_execution=False,  # Use sync mode for simpler debugging
        max_concurrent_evaluations=2,
        # Output
        output_dir=tmp_path / "minimal_output",
        verbose=False,  # Reduce noise in test output
        # Task
        task_spec=sample_task_spec,
    )


@pytest.fixture
def realistic_config(sample_task_spec, tmp_path):
    """
    Provide realistic configuration matching actual usage.

    Uses the same numbers as production:
    - 15 initial prompts
    - 7 quick tests
    - 50 rigorous tests
    - Top 5 advance to rigorous
    - Top 3 go to refinement
    - Max 10 refinement iterations
    """
    return OptimizerConfig(
        # Stage 1: Quick filter
        num_initial_prompts=15,
        quick_test_distribution=TestDistribution(
            core=2, edge=2, boundary=1, adversarial=1, consistency=1, format=0
        ),
        top_k_advance=5,
        # Stage 2: Rigorous testing
        rigorous_test_distribution=TestDistribution(
            core=20, edge=10, boundary=10, adversarial=5, consistency=3, format=2
        ),
        top_m_refine=3,
        # Stage 3: Refinement
        max_iterations_per_track=10,
        convergence_threshold=0.02,
        early_stopping_patience=2,
        # LLM configs
        generator_llm=LLMConfig(model="gpt-4o", temperature=0.8),
        test_designer_llm=LLMConfig(model="gpt-4o", temperature=0.7),
        evaluator_llm=LLMConfig(model="gpt-4o", temperature=0.3),
        refiner_llm=LLMConfig(model="gpt-4o", temperature=0.7),
        # Execution
        parallel_execution=False,
        max_concurrent_evaluations=5,
        # Output
        output_dir=tmp_path / "realistic_output",
        verbose=False,
        # Task
        task_spec=sample_task_spec,
    )


@pytest.fixture
def parallel_config(minimal_config):
    """
    Provide config with parallel execution enabled.

    Useful for testing async/parallel code paths.
    """
    config = minimal_config.model_copy()
    config.parallel_execution = True
    return config


@pytest.fixture
def early_stopping_config(minimal_config):
    """
    Provide config tuned for testing early stopping.

    Very low convergence threshold and low patience to trigger stopping.
    """
    config = minimal_config.model_copy()
    config.convergence_threshold = 0.01  # 1% improvement required
    config.early_stopping_patience = 1  # Stop after 1 iteration without improvement
    config.max_iterations_per_track=5
    return config


@pytest.fixture
def config_with_original_prompt(minimal_config, sample_task_spec_with_original_prompt):
    """Provide config with an original prompt for comparison testing."""
    config = minimal_config.model_copy()
    config.task_spec = sample_task_spec_with_original_prompt
    return config
