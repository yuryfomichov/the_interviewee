"""Data types and models for prompt optimization."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class TaskSpec(BaseModel):
    """Specification of the task to optimize prompts for."""

    task_description: str = Field(description="What the AI should do")
    behavioral_specs: str = Field(description="How it should work, rules, constraints, tone")
    validation_rules: list[str] = Field(description="List of rules to validate behavior")
    current_prompt: str | None = Field(
        default=None, description="Optional existing prompt to improve upon"
    )


class TestCase(BaseModel):
    """A single test case for evaluating prompt performance."""

    id: str = Field(description="Unique identifier for the test case")
    input_message: str = Field(description="User message to test with")
    expected_behavior: str = Field(description="What the AI should do for this input")
    category: Literal["core", "edge", "boundary", "adversarial", "consistency", "format"] = Field(
        description="Category of test case"
    )


class EvaluationScore(BaseModel):
    """Evaluation scores for a single response."""

    functionality: int = Field(ge=0, le=10, description="Does it accomplish the task?")
    safety: int = Field(ge=0, le=10, description="Handles boundaries properly?")
    consistency: int = Field(ge=0, le=10, description="Appropriate tone/style/behavior?")
    edge_case_handling: int = Field(ge=0, le=10, description="Handles this test well?")
    reasoning: str = Field(description="Brief explanation of scores")
    overall: float = Field(ge=0, le=10, description="Weighted overall score")

    @classmethod
    def calculate_overall(
        cls,
        functionality: int,
        safety: int,
        consistency: int,
        edge_case_handling: int,
        reasoning: str,
        weights: dict[str, float],
    ) -> "EvaluationScore":
        """Calculate overall score with custom weights."""
        overall = (
            functionality * weights["functionality"]
            + safety * weights["safety"]
            + consistency * weights["consistency"]
            + edge_case_handling * weights["edge_case_handling"]
        )
        return cls(
            functionality=functionality,
            safety=safety,
            consistency=consistency,
            edge_case_handling=edge_case_handling,
            reasoning=reasoning,
            overall=overall,
        )


class TestResult(BaseModel):
    """Result of testing a prompt with a single test case."""

    test_case_id: str
    prompt_id: str
    model_response: str
    evaluation: EvaluationScore
    timestamp: datetime = Field(default_factory=datetime.now)


class PromptCandidate(BaseModel):
    """A candidate system prompt at a specific stage."""

    id: str = Field(description="Unique identifier")
    prompt_text: str = Field(description="The actual system prompt text")
    stage: Literal["initial", "quick_filter", "rigorous", "refined"] = Field(
        description="Which stage this prompt is from"
    )
    strategy: str | None = Field(
        default=None, description="Generation strategy (e.g., 'structured', 'conversational')"
    )
    average_score: float | None = Field(
        default=None, description="Average evaluation score across all tests"
    )
    iteration: int = Field(default=0, description="Refinement iteration number")
    track_id: int | None = Field(default=None, description="Refinement track number (0-2)")
    created_at: datetime = Field(default_factory=datetime.now)


class RefinementTrackResult(BaseModel):
    """Results from a single refinement track."""

    track_id: int
    initial_prompt: PromptCandidate
    final_prompt: PromptCandidate
    iterations: list[PromptCandidate]
    score_progression: list[float]
    improvement: float  # Final score - initial score


class OptimizationResult(BaseModel):
    """Final results from the entire optimization process."""

    run_id: int | None = Field(default=None, description="Identifier for the storage run")
    best_prompt: PromptCandidate = Field(description="Champion prompt")
    all_tracks: list[RefinementTrackResult] = Field(description="Results from all 3 tracks")
    initial_prompts: list[PromptCandidate] = Field(description="All 15 initial prompts")
    stage1_top5: list[PromptCandidate] = Field(description="Top 5 from quick filter")
    stage2_top3: list[PromptCandidate] = Field(description="Top 3 from rigorous testing")
    total_tests_run: int = Field(description="Total number of test evaluations")
    total_time_seconds: float = Field(description="Total optimization time")
    quick_tests: list[TestCase] = Field(description="Quick evaluation test cases")
    rigorous_tests: list[TestCase] = Field(description="Rigorous evaluation test cases")
