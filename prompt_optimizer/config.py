"""Configuration for prompt optimizer with per-agent LLM settings."""

from pydantic import BaseModel, Field

from prompt_optimizer.types import TaskSpec


class LLMConfig(BaseModel):
    """Configuration for a single LLM instance.

    Note: OpenAI Agents SDK currently doesn't support per-agent temperature settings.
    Temperature is controlled globally via OpenAI API client configuration.
    The temperature field is kept for future compatibility and documentation.
    """

    model: str = Field(description="Model name (e.g., 'gpt-4o', 'claude-sonnet-4-5')")
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Desired sampling temperature (not currently used by Agents SDK)",
    )
    max_tokens: int | None = Field(default=None, description="Maximum tokens to generate")


class TestDistribution(BaseModel):
    """Distribution of test cases across different categories."""

    core: int = Field(default=20, ge=0, description="Tests of primary task capabilities")
    edge: int = Field(default=10, ge=0, description="Unusual inputs, corner cases, rare scenarios")
    boundary: int = Field(default=10, ge=0, description="What it should/shouldn't do, limits")
    adversarial: int = Field(
        default=5, ge=0, description="Attempts to break behavior, trick responses"
    )
    consistency: int = Field(default=3, ge=0, description="Style, tone, format consistency checks")
    format: int = Field(default=2, ge=0, description="Citation requirements, structure adherence")

    @property
    def total(self) -> int:
        """Calculate total number of tests."""
        return (
            self.core
            + self.edge
            + self.boundary
            + self.adversarial
            + self.consistency
            + self.format
        )

    def to_dict(self) -> dict[str, int]:
        """Convert to dictionary for compatibility."""
        return {
            "core": self.core,
            "edge": self.edge,
            "boundary": self.boundary,
            "adversarial": self.adversarial,
            "consistency": self.consistency,
            "format": self.format,
        }


class OptimizerConfig(BaseModel):
    """Configuration for the entire optimization pipeline."""

    # Stage 1: Quick Filter
    num_initial_prompts: int = Field(
        default=15, description="Number of diverse prompts to generate"
    )
    quick_test_distribution: TestDistribution = Field(
        default_factory=lambda: TestDistribution(
            core=2, edge=2, boundary=1, adversarial=1, consistency=1, format=0
        ),
        description="Distribution of quick evaluation tests",
    )
    top_k_advance: int = Field(default=5, description="Top K prompts advance to Stage 2")

    # Stage 2: Rigorous Testing
    rigorous_test_distribution: TestDistribution = Field(
        default_factory=TestDistribution,
        description="Distribution of comprehensive evaluation tests",
    )
    top_m_refine: int = Field(default=3, description="Top M prompts advance to Stage 3")

    # Stage 3: Refinement
    max_iterations_per_track: int = Field(
        default=10, description="Maximum refinement iterations per track"
    )
    convergence_threshold: float = Field(
        default=0.02, description="Minimum improvement to continue (2%)"
    )
    early_stopping_patience: int = Field(
        default=2, description="Stop after N iterations without improvement"
    )

    # Scoring weights
    scoring_weights: dict[str, float] = Field(
        default={
            "functionality": 0.4,
            "safety": 0.3,
            "consistency": 0.2,
            "edge_case_handling": 0.1,
        },
        description="Weights for score dimensions",
    )

    # LLM Configuration per Agent (NEW: Configurable per agent!)
    generator_llm: LLMConfig = Field(
        default=LLMConfig(model="gpt-4o", temperature=0.8),
        description="LLM for prompt generation (higher temp for creativity)",
    )
    test_designer_llm: LLMConfig = Field(
        default=LLMConfig(model="gpt-4o", temperature=0.7),
        description="LLM for test case generation",
    )
    evaluator_llm: LLMConfig = Field(
        default=LLMConfig(model="gpt-4o", temperature=0.3),
        description="LLM for scoring (lower temp for consistency)",
    )
    refiner_llm: LLMConfig = Field(
        default=LLMConfig(model="gpt-4o", temperature=0.7),
        description="LLM for prompt refinement",
    )

    # Storage
    storage_path: str = Field(
        default="prompt_optimizer/data/optimizer.db", description="SQLite database path"
    )

    # Execution mode
    parallel_execution: bool = Field(
        default=False,
        description="Run stages in parallel mode (True) or sequential/sync mode (False)",
    )

    # Progress reporting
    verbose: bool = Field(default=True, description="Print progress updates")

    # API Configuration
    openai_api_key: str | None = Field(
        default=None, description="OpenAI API key (if None, uses OPENAI_API_KEY env var)"
    )

    # Task specification
    task_spec: TaskSpec = Field(
        ...,
        description="Task specification defining behavioral requirements for the target assistant.",
    )

    @property
    def num_quick_tests(self) -> int:
        """Calculate total number of quick tests from distribution."""
        return self.quick_test_distribution.total

    @property
    def num_rigorous_tests(self) -> int:
        """Calculate total number of rigorous tests from distribution."""
        return self.rigorous_test_distribution.total
