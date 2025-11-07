"""Test full end-to-end pipeline execution."""

import pytest

from prompt_optimizer.optimizer.orchestrator import PromptOptimizer
from prompt_optimizer.schemas import OptimizationResult


@pytest.mark.asyncio
async def test_minimal_pipeline_completes(
    minimal_config, dummy_connector, mock_agents, test_database
):
    """
    Test that the full 10-stage pipeline executes successfully with minimal config.

    This is a smoke test to ensure basic pipeline functionality works.
    """
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    result = await optimizer.optimize()

    # Verify result structure
    assert result is not None
    assert isinstance(result, OptimizationResult)
    assert result.best_prompt is not None
    assert result.best_prompt.prompt_text is not None
    assert len(result.best_prompt.prompt_text) > 0

    # Verify refinement tracks
    assert len(result.all_tracks) == minimal_config.top_m_refine
    for track in result.all_tracks:
        assert track.initial_prompt is not None
        assert track.final_prompt is not None
        assert len(track.iterations) >= 0

    # Verify test counts
    assert len(result.quick_tests) == minimal_config.num_quick_tests
    assert len(result.rigorous_tests) == minimal_config.num_rigorous_tests

    # Verify prompts at each stage
    assert len(result.initial_prompts) == minimal_config.num_initial_prompts
    assert len(result.top_k_prompts) == minimal_config.top_k_advance
    assert len(result.top_m_prompts) == minimal_config.top_m_refine

    # Verify test execution count
    assert result.total_tests_run > 0

    # Verify timing
    assert result.total_time_seconds > 0

    # Verify output directory was set
    assert result.output_dir is not None
    assert result.run_id is not None


@pytest.mark.asyncio
async def test_realistic_pipeline_completes(
    realistic_config, dummy_connector, mock_agents, test_database
):
    """
    Test that pipeline works with realistic production-like configuration.

    Uses full-scale numbers: 15 prompts, 50 rigorous tests, etc.
    """
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=realistic_config, database=test_database
    )

    result = await optimizer.optimize()

    # Verify basic structure
    assert result is not None
    assert result.best_prompt is not None

    # Verify counts match config
    assert len(result.initial_prompts) == realistic_config.num_initial_prompts
    assert len(result.top_k_prompts) == realistic_config.top_k_advance
    assert len(result.top_m_prompts) == realistic_config.top_m_refine
    assert len(result.all_tracks) == realistic_config.top_m_refine

    # Verify test suite sizes
    assert len(result.quick_tests) == realistic_config.num_quick_tests
    assert len(result.rigorous_tests) == realistic_config.num_rigorous_tests

    # Verify all stages produced valid outputs
    for prompt in result.initial_prompts:
        assert prompt.stage == "initial"
        assert prompt.prompt_text is not None

    for prompt in result.top_k_prompts:
        assert prompt.stage == "quick_filter"
        assert prompt.quick_score is not None

    for prompt in result.top_m_prompts:
        assert prompt.stage == "rigorous"
        assert prompt.rigorous_score is not None


@pytest.mark.asyncio
async def test_parallel_execution_mode(
    parallel_config, dummy_connector, mock_agents, test_database
):
    """
    Test that pipeline works in parallel execution mode.

    Verifies async/parallel code paths execute correctly.
    """
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=parallel_config, database=test_database
    )

    result = await optimizer.optimize()

    # Should produce same structure as sync mode
    assert result is not None
    assert result.best_prompt is not None
    assert len(result.all_tracks) == parallel_config.top_m_refine
    assert len(result.initial_prompts) == parallel_config.num_initial_prompts


@pytest.mark.asyncio
async def test_pipeline_with_original_prompt(
    config_with_original_prompt, dummy_connector, mock_agents, test_database
):
    """
    Test that pipeline correctly handles and tracks the original prompt.

    The original prompt should:
    - Be included in initial prompts
    - Be evaluated on quick tests
    - Be evaluated on rigorous tests for comparison
    - Not advance if it doesn't score well
    """
    optimizer = PromptOptimizer(
        model_client=dummy_connector,
        config=config_with_original_prompt,
        database=test_database,
    )

    result = await optimizer.optimize()

    # Verify original prompt is tracked
    assert result.original_system_prompt is not None
    assert result.original_system_prompt.is_original_system_prompt is True
    assert result.original_system_prompt.strategy == "original_system_prompt"

    # Verify original prompt was evaluated on rigorous tests
    assert result.original_system_prompt_rigorous_score is not None
    assert result.original_system_prompt_rigorous_score > 0

    # Verify test results exist for original prompt
    assert len(result.original_system_prompt_test_results) > 0


@pytest.mark.asyncio
async def test_champion_has_test_results(
    minimal_config, dummy_connector, mock_agents, test_database
):
    """
    Test that the champion prompt has associated test results.

    The champion should have rigorous test results stored.
    """
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    result = await optimizer.optimize()

    # Champion should have test results
    assert len(result.champion_test_results) > 0

    # All test results should reference the champion prompt
    for test_result in result.champion_test_results:
        assert test_result.prompt_id == result.best_prompt.id


@pytest.mark.asyncio
async def test_database_state_after_pipeline(
    minimal_config, dummy_connector, mock_agents, test_database
):
    """
    Test that database is correctly populated after pipeline completes.

    Verifies that all stages left appropriate records in the database.
    """
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    result = await optimizer.optimize()
    run_id = result.run_id

    # Get database session to inspect state
    session = test_database.get_session()

    try:
        from prompt_optimizer.storage.models import Evaluation, Prompt, OptimizationRun, TestCase

        # Verify run exists
        run = session.query(OptimizationRun).filter_by(id=run_id).first()
        assert run is not None
        assert run.task_description is not None

        # Verify prompts exist - check using result object
        # (database stages show final state where prompts have progressed)
        assert len(result.initial_prompts) == minimal_config.num_initial_prompts
        assert len(result.top_k_prompts) == minimal_config.top_k_advance
        assert len(result.top_m_prompts) == minimal_config.top_m_refine
        assert len(result.all_tracks) > 0

        # Verify test cases exist
        quick_tests = session.query(TestCase).filter_by(run_id=run_id, stage="quick").all()
        assert len(quick_tests) == minimal_config.num_quick_tests

        rigorous_tests = session.query(TestCase).filter_by(run_id=run_id, stage="rigorous").all()
        assert len(rigorous_tests) == minimal_config.num_rigorous_tests

        # Verify evaluations exist
        evaluations = session.query(Evaluation).filter_by(run_id=run_id).all()
        assert len(evaluations) > 0

    finally:
        session.close()


@pytest.mark.asyncio
async def test_multiple_runs_isolated(
    minimal_config, dummy_connector, mock_agents, test_database
):
    """
    Test that multiple optimization runs are properly isolated in the database.

    Each run should have its own run_id and separate data.
    """
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    # Run pipeline twice
    result1 = await optimizer.optimize()
    result2 = await optimizer.optimize()

    # Runs should have different IDs
    assert result1.run_id != result2.run_id

    # Verify both runs exist in database
    session = test_database.get_session()
    try:
        from prompt_optimizer.storage.models import OptimizationRun

        run1 = session.query(OptimizationRun).filter_by(id=result1.run_id).first()
        run2 = session.query(OptimizationRun).filter_by(id=result2.run_id).first()

        assert run1 is not None
        assert run2 is not None
        assert run1.id != run2.id

    finally:
        session.close()
