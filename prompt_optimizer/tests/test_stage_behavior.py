"""Test individual stage behavior and logic."""

import pytest

from prompt_optimizer.optimizer.orchestrator import PromptOptimizer


@pytest.mark.asyncio
async def test_generate_prompts_stage_creates_correct_count(
    minimal_config, dummy_connector, mock_agents, test_database
):
    """Test that GeneratePromptsStage creates exactly N prompts."""
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    result = await optimizer.optimize()

    # Check initial prompts from result (they're stored there after pipeline completes)
    assert len(result.initial_prompts) == minimal_config.num_initial_prompts

    # All should have prompt_text
    for prompt in result.initial_prompts:
        assert prompt.prompt_text is not None
        assert len(prompt.prompt_text) > 0
        assert prompt.strategy is not None


@pytest.mark.asyncio
async def test_generate_tests_stage_creates_test_suites(
    minimal_config, dummy_connector, mock_agents, test_database
):
    """Test that GenerateTestsStage creates both quick and rigorous test suites."""
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    result = await optimizer.optimize()

    # Check tests from result (they're stored there after pipeline completes)
    assert len(result.quick_tests) == minimal_config.num_quick_tests
    assert len(result.rigorous_tests) == minimal_config.num_rigorous_tests

    # Verify test structure
    for test in result.quick_tests + result.rigorous_tests:
        assert test.input_message is not None
        assert test.expected_behavior is not None
        assert test.category in [
            "core",
            "edge",
            "boundary",
            "adversarial",
            "consistency",
            "format",
        ]


@pytest.mark.asyncio
async def test_evaluate_prompts_stage_creates_evaluations(
    minimal_config, dummy_connector, mock_agents, test_database
):
    """Test that EvaluatePromptsStage creates evaluation records."""
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    result = await optimizer.optimize()
    session = test_database.get_session()

    try:
        from prompt_optimizer.storage.models import Evaluation

        # Should have evaluations for quick stage
        quick_evaluations = session.query(Evaluation).filter(
            Evaluation.run_id == result.run_id
        ).all()

        assert len(quick_evaluations) > 0

        # Verify evaluation structure
        for evaluation in quick_evaluations[:5]:  # Check first 5
            assert evaluation.prompt_id is not None
            assert evaluation.test_case_id is not None
            assert evaluation.model_response is not None
            assert evaluation.overall_score is not None
            assert 0 <= evaluation.overall_score <= 10
            assert evaluation.functionality is not None
            assert evaluation.safety is not None
            assert evaluation.consistency is not None
            assert evaluation.edge_case_handling is not None

    finally:
        session.close()


@pytest.mark.asyncio
async def test_select_top_prompts_stage_selects_by_score(
    minimal_config, dummy_connector, mock_agents, test_database
):
    """Test that SelectTopPromptsStage correctly selects top performers."""
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    result = await optimizer.optimize()

    # Use result object (database stages show final state where prompts have progressed)
    # Get all initial prompts sorted by quick_score
    all_initial = sorted(
        [p for p in result.initial_prompts if p.quick_score is not None],
        key=lambda p: p.quick_score,
        reverse=True
    )

    # Get selected prompts (top K)
    selected = sorted(
        result.top_k_prompts,
        key=lambda p: p.quick_score,
        reverse=True
    )

    # Selected prompts should be top K
    assert len(selected) == minimal_config.top_k_advance

    # Scores should match top K from all prompts
    expected_top_scores = [p.quick_score for p in all_initial[: minimal_config.top_k_advance]]
    actual_scores = [p.quick_score for p in selected]

    assert actual_scores == expected_top_scores


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
async def test_refinement_stage_creates_tracks(
    minimal_config, dummy_connector, mock_agents, test_database
):
    """Test that RefinementStage creates proper refinement tracks."""
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    result = await optimizer.optimize()
    session = test_database.get_session()

    try:
        from prompt_optimizer.storage.models import Prompt

        # Should have refined prompts
        refined_prompts = session.query(Prompt).filter_by(
            run_id=result.run_id, stage="refined"
        ).all()

        assert len(refined_prompts) > 0

        # Should have track_ids
        track_ids = set(p.track_id for p in refined_prompts if p.track_id is not None)
        assert len(track_ids) == minimal_config.top_m_refine

        # Verify track_ids are in expected range
        expected_track_ids = set(range(minimal_config.top_m_refine))
        assert track_ids == expected_track_ids

        # Verify parent-child relationships
        for prompt in refined_prompts:
            if prompt.iteration > 0:
                assert prompt.parent_prompt_id is not None

    finally:
        session.close()


@pytest.mark.asyncio
async def test_reporting_stage_creates_complete_result(
    minimal_config, dummy_connector, mock_agents, test_database
):
    """Test that ReportingStage creates complete OptimizationResult."""
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    result = await optimizer.optimize()

    # Verify all fields are populated
    assert result.run_id is not None
    assert result.output_dir is not None
    assert result.best_prompt is not None
    assert len(result.all_tracks) > 0
    assert len(result.initial_prompts) > 0
    assert len(result.top_k_prompts) > 0
    assert len(result.top_m_prompts) > 0
    assert result.total_tests_run > 0
    assert result.total_time_seconds > 0
    assert len(result.quick_tests) > 0
    assert len(result.rigorous_tests) > 0
    assert len(result.champion_test_results) > 0

    # Verify refinement tracks structure
    for track in result.all_tracks:
        assert track.track_id is not None
        assert track.initial_prompt is not None
        assert track.final_prompt is not None
        assert isinstance(track.iterations, list)
        assert isinstance(track.score_progression, list)
        assert track.improvement is not None


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
