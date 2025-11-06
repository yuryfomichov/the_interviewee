"""Convert between Pydantic models and SQLAlchemy models."""

import json
from datetime import datetime

from prompt_optimizer.storage.models import (
    Evaluation,
    Prompt,
    TestCase as DbTestCase,
    WeaknessAnalysis,
)
from prompt_optimizer.types import (
    EvaluationScore,
    PromptCandidate,
    TestCase as PydanticTestCase,
    TestResult,
    WeaknessAnalysis as PydanticWeakness,
)


class PromptConverter:
    """Convert between Pydantic PromptCandidate and SQLAlchemy Prompt."""

    @staticmethod
    def to_db(candidate: PromptCandidate, run_id: int) -> Prompt:
        """Convert Pydantic PromptCandidate to SQLAlchemy Prompt."""
        return Prompt(
            id=candidate.id,
            run_id=run_id,
            prompt_text=candidate.prompt_text,
            stage=candidate.stage,
            strategy=candidate.strategy,
            average_score=candidate.average_score,
            quick_score=candidate.quick_score,
            rigorous_score=candidate.rigorous_score,
            iteration=candidate.iteration,
            track_id=candidate.track_id,
            parent_prompt_id=None,  # Set separately during refinement
            is_original_system_prompt=candidate.is_original_system_prompt,
            created_at=candidate.created_at,
        )

    @staticmethod
    def from_db(prompt: Prompt) -> PromptCandidate:
        """Convert SQLAlchemy Prompt to Pydantic PromptCandidate."""
        return PromptCandidate(
            id=prompt.id,
            prompt_text=prompt.prompt_text,
            stage=prompt.stage,
            strategy=prompt.strategy,
            average_score=prompt.average_score,
            quick_score=prompt.quick_score,
            rigorous_score=prompt.rigorous_score,
            iteration=prompt.iteration,
            track_id=prompt.track_id,
            is_original_system_prompt=prompt.is_original_system_prompt,
            created_at=prompt.created_at,
        )


class TestCaseConverter:
    """Convert between Pydantic TestCase and SQLAlchemy TestCase."""

    @staticmethod
    def to_db(test: PydanticTestCase, run_id: int, stage: str) -> DbTestCase:
        """Convert Pydantic TestCase to SQLAlchemy TestCase."""
        return DbTestCase(
            id=test.id,
            run_id=run_id,
            input_message=test.input_message,
            expected_behavior=test.expected_behavior,
            category=test.category,
            stage=stage,
        )

    @staticmethod
    def from_db(test: DbTestCase) -> PydanticTestCase:
        """Convert SQLAlchemy TestCase to Pydantic TestCase."""
        return PydanticTestCase(
            id=test.id,
            input_message=test.input_message,
            expected_behavior=test.expected_behavior,
            category=test.category,
        )


class EvaluationConverter:
    """Convert between Pydantic TestResult and SQLAlchemy Evaluation."""

    @staticmethod
    def to_db(result: TestResult, run_id: int) -> Evaluation:
        """Convert Pydantic TestResult to SQLAlchemy Evaluation."""
        return Evaluation(
            run_id=run_id,
            test_case_id=result.test_case_id,
            prompt_id=result.prompt_id,
            model_response=result.model_response,
            functionality=result.evaluation.functionality,
            safety=result.evaluation.safety,
            consistency=result.evaluation.consistency,
            edge_case_handling=result.evaluation.edge_case_handling,
            reasoning=result.evaluation.reasoning,
            overall_score=result.evaluation.overall,
            timestamp=result.timestamp,
        )

    @staticmethod
    def from_db(evaluation: Evaluation) -> TestResult:
        """Convert SQLAlchemy Evaluation to Pydantic TestResult."""
        return TestResult(
            test_case_id=evaluation.test_case_id,
            prompt_id=evaluation.prompt_id,
            model_response=evaluation.model_response,
            evaluation=EvaluationScore(
                functionality=evaluation.functionality,
                safety=evaluation.safety,
                consistency=evaluation.consistency,
                edge_case_handling=evaluation.edge_case_handling,
                reasoning=evaluation.reasoning,
                overall=evaluation.overall_score,
            ),
            timestamp=evaluation.timestamp,
        )


class WeaknessAnalysisConverter:
    """Convert between Pydantic WeaknessAnalysis and SQLAlchemy WeaknessAnalysis."""

    @staticmethod
    def to_db(weakness: PydanticWeakness, prompt_id: str) -> WeaknessAnalysis:
        """Convert Pydantic WeaknessAnalysis to SQLAlchemy WeaknessAnalysis."""
        return WeaknessAnalysis(
            prompt_id=prompt_id,
            iteration=weakness.iteration,
            description=weakness.description,
            failed_test_ids=json.dumps(weakness.failed_test_ids),
            failed_test_descriptions=json.dumps(weakness.failed_test_descriptions),
        )

    @staticmethod
    def from_db(weakness: WeaknessAnalysis) -> PydanticWeakness:
        """Convert SQLAlchemy WeaknessAnalysis to Pydantic WeaknessAnalysis."""
        return PydanticWeakness(
            iteration=weakness.iteration,
            description=weakness.description,
            failed_test_ids=json.loads(weakness.failed_test_ids),
            failed_test_descriptions=json.loads(weakness.failed_test_descriptions),
        )
