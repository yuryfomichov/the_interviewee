"""Run context for passing data between optimization stages."""

from pydantic import BaseModel, Field

from prompt_optimizer.types import (
    OptimizationResult,
    PromptCandidate,
    RefinementTrackResult,
    TaskSpec,
    TestCase,
)


class RunContext(BaseModel):
    """Context object passed between optimization stages.

    Each stage reads from and writes to this context, making the pipeline
    data flow explicit and type-safe.
    """

    # Input: Task specification
    task_spec: TaskSpec = Field(description="Task specification for optimization")

    # Output directory for saving intermediate results
    output_dir: str | None = Field(
        default=None, description="Directory for saving intermediate reports during optimization"
    )

    # Run metadata
    run_id: int | None = Field(default=None, description="Optimization run ID")
    start_time: float | None = Field(default=None, description="When optimization started")

    # Stage 1 output: Generated prompts
    initial_prompts: list[PromptCandidate] = Field(
        default_factory=list, description="All initially generated prompts"
    )

    # Stage 2 output: Quick tests
    quick_tests: list[TestCase] = Field(
        default_factory=list, description="Quick evaluation test cases"
    )

    # Stage 3 output: Evaluated prompts (implicit, uses initial_prompts with updated scores)

    # Stage 4 output: Top K prompts
    top_k_prompts: list[PromptCandidate] = Field(
        default_factory=list, description="Top K prompts after quick filter"
    )

    # Stage 5 output: Rigorous tests
    rigorous_tests: list[TestCase] = Field(
        default_factory=list, description="Rigorous evaluation test cases"
    )

    # Stage 6 output: Evaluated top K (implicit, uses top_k_prompts with updated scores)

    # Stage 7 output: Top M prompts
    top_m_prompts: list[PromptCandidate] = Field(
        default_factory=list, description="Top M prompts after rigorous testing"
    )

    # Stage 8 output: Refinement results
    refinement_tracks: list[RefinementTrackResult] = Field(
        default_factory=list, description="Results from parallel refinement tracks"
    )

    # Stage 9 output: Final optimization result
    optimization_result: OptimizationResult | None = Field(
        default=None, description="Final optimization result with complete analysis"
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True
