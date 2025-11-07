"""Test individual stage behavior and logic."""

import pytest

from prompt_optimizer.optimizer.orchestrator import PromptOptimizer


@pytest.mark.asyncio
async def test_evaluation_scores_are_consistent(
    minimal_config, dummy_connector, mock_agents, test_database
):
    """Test that evaluation scores are calculated consistently."""
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    result = await optimizer.optimize()
    session = test_database.get_session()

    try:
        from prompt_optimizer.storage.models import Evaluation

        evaluations = session.query(Evaluation).filter_by(run_id=result.run_id).all()

        assert len(evaluations) > 0

        for evaluation in evaluations:
            # Verify overall score is within bounds
            assert 0 <= evaluation.overall_score <= 10

            # Verify component scores are within bounds
            assert 0 <= evaluation.functionality <= 10
            assert 0 <= evaluation.safety <= 10
            assert 0 <= evaluation.consistency <= 10
            assert 0 <= evaluation.edge_case_handling <= 10

            # Verify overall is roughly weighted average (using default weights)
            weights = minimal_config.scoring_weights
            expected_overall = (
                evaluation.functionality * weights["functionality"]
                + evaluation.safety * weights["safety"]
                + evaluation.consistency * weights["consistency"]
                + evaluation.edge_case_handling * weights["edge_case_handling"]
            )

            # Allow small floating point difference
            assert abs(evaluation.overall_score - expected_overall) < 0.1

    finally:
        session.close()


@pytest.mark.asyncio
async def test_test_distribution_respected(
    realistic_config, dummy_connector, mock_agents, test_database
):
    """Test that test case distribution matches configuration."""
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=realistic_config, database=test_database
    )

    result = await optimizer.optimize()
    session = test_database.get_session()

    try:
        from prompt_optimizer.storage.models import TestCase

        # Check rigorous test distribution
        rigorous_tests = session.query(TestCase).filter_by(
            run_id=result.run_id, stage="rigorous"
        ).all()

        # Count by category
        category_counts = {}
        for test in rigorous_tests:
            category_counts[test.category] = category_counts.get(test.category, 0) + 1

        # Verify counts match config distribution
        expected = realistic_config.rigorous_test_distribution
        assert category_counts.get("core", 0) == expected.core
        assert category_counts.get("edge", 0) == expected.edge
        assert category_counts.get("boundary", 0) == expected.boundary
        assert category_counts.get("adversarial", 0) == expected.adversarial
        assert category_counts.get("consistency", 0) == expected.consistency
        assert category_counts.get("format", 0) == expected.format

    finally:
        session.close()
