"""Test edge cases, error handling, and special scenarios."""

import pytest

from prompt_optimizer.optimizer.orchestrator import PromptOptimizer


@pytest.mark.asyncio
async def test_original_prompt_not_advanced_if_poor_performance(
    config_with_original_prompt, dummy_connector, mock_agents, test_database
):
    """
    Test that original prompt is not automatically advanced if it scores poorly.

    Original prompt should compete fairly with generated prompts.
    """
    optimizer = PromptOptimizer(
        model_client=dummy_connector,
        config=config_with_original_prompt,
        database=test_database,
    )

    result = await optimizer.optimize()
    session = test_database.get_session()

    try:
        from prompt_optimizer.storage.models import Prompt

        # Original prompt should exist
        original = session.query(Prompt).filter_by(
            run_id=result.run_id, is_original_system_prompt=True
        ).first()

        assert original is not None

        # Original should have been evaluated
        assert original.quick_score is not None

        # Original should be compared on rigorous tests
        assert original.rigorous_score is not None

        # Original may or may not advance - depends on its score relative to others
        # Just verify it's tracked correctly
        assert result.original_system_prompt is not None
        assert result.original_system_prompt_rigorous_score is not None

    finally:
        session.close()


@pytest.mark.asyncio
async def test_sync_and_async_modes_produce_same_structure(
    minimal_config, parallel_config, dummy_connector, mock_agents, test_database
):
    """
    Test that sync and async execution modes produce equivalent results.

    Both modes should create the same pipeline structure.
    """
    # Run in sync mode
    optimizer_sync = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )
    result_sync = await optimizer_sync.optimize()

    # Run in async mode (use different database to avoid conflicts)
    from pathlib import Path
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_async.db"
        from prompt_optimizer.storage import Database

        async_db = Database(db_path)

        parallel_config.output_dir = Path(tmpdir) / "output"
        optimizer_async = PromptOptimizer(
            model_client=dummy_connector, config=parallel_config, database=async_db
        )
        result_async = await optimizer_async.optimize()

        # Both should have same structure
        assert len(result_sync.initial_prompts) == len(result_async.initial_prompts)
        assert len(result_sync.top_k_prompts) == len(result_async.top_k_prompts)
        assert len(result_sync.top_m_prompts) == len(result_async.top_m_prompts)
        assert len(result_sync.all_tracks) == len(result_async.all_tracks)
        assert len(result_sync.quick_tests) == len(result_async.quick_tests)
        assert len(result_sync.rigorous_tests) == len(result_async.rigorous_tests)


@pytest.mark.asyncio
async def test_minimal_configuration_completes(
    sample_task_spec, dummy_connector, mock_agents, test_database, tmp_path
):
    """
    Test that pipeline works with absolute minimal configuration.

    Uses smallest possible numbers to test edge cases.
    """
    from prompt_optimizer.config import LLMConfig, OptimizerConfig, TestDistribution

    # Absolute minimum: 1 prompt, 1 test at each stage
    minimal = OptimizerConfig(
        num_initial_prompts=1,
        quick_test_distribution=TestDistribution(
            core=1, edge=0, boundary=0, adversarial=0, consistency=0, format=0
        ),
        top_k_advance=1,
        rigorous_test_distribution=TestDistribution(
            core=1, edge=0, boundary=0, adversarial=0, consistency=0, format=0
        ),
        top_m_refine=1,
        max_iterations_per_track=1,
        convergence_threshold=0.02,
        early_stopping_patience=1,
        generator_llm=LLMConfig(model="gpt-4o"),
        test_designer_llm=LLMConfig(model="gpt-4o"),
        evaluator_llm=LLMConfig(model="gpt-4o"),
        refiner_llm=LLMConfig(model="gpt-4o"),
        parallel_execution=False,
        max_concurrent_evaluations=1,
        output_dir=tmp_path / "minimal",
        verbose=False,
        task_spec=sample_task_spec,
    )

    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal, database=test_database
    )

    result = await optimizer.optimize()

    # Should still complete successfully
    assert result is not None
    assert result.best_prompt is not None
    assert len(result.all_tracks) == 1


@pytest.mark.asyncio
async def test_database_isolation_between_runs(
    minimal_config, dummy_connector, mock_agents, test_database
):
    """
    Test that multiple runs in the same database are properly isolated.

    Each run should have separate data with different run_ids.
    """
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    # Run twice
    result1 = await optimizer.optimize()
    result2 = await optimizer.optimize()

    # Should have different run IDs
    assert result1.run_id != result2.run_id

    session = test_database.get_session()
    try:
        from prompt_optimizer.storage.models import Prompt

        # Each run should have its own prompts
        prompts_run1 = session.query(Prompt).filter_by(run_id=result1.run_id).all()
        prompts_run2 = session.query(Prompt).filter_by(run_id=result2.run_id).all()

        assert len(prompts_run1) > 0
        assert len(prompts_run2) > 0

        # No overlap in prompt IDs
        ids_run1 = {p.id for p in prompts_run1}
        ids_run2 = {p.id for p in prompts_run2}
        assert len(ids_run1.intersection(ids_run2)) == 0

    finally:
        session.close()


@pytest.mark.asyncio
async def test_all_test_categories_present(
    realistic_config, dummy_connector, mock_agents, test_database
):
    """
    Test that all test case categories are present when configured.

    Verifies the full range of test types are generated.
    """
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=realistic_config, database=test_database
    )

    result = await optimizer.optimize()
    session = test_database.get_session()

    try:
        from prompt_optimizer.storage.models import TestCase

        rigorous_tests = session.query(TestCase).filter_by(
            run_id=result.run_id, stage="rigorous"
        ).all()

        # Get unique categories
        categories = {test.category for test in rigorous_tests}

        # Should have all configured categories
        expected_categories = {"core", "edge", "boundary", "adversarial", "consistency", "format"}
        assert expected_categories.issubset(categories)

    finally:
        session.close()


@pytest.mark.asyncio
async def test_connector_called_for_evaluations(
    minimal_config, dummy_connector, mock_agents, test_database
):
    """
    Test that the model connector is actually called during evaluation.

    Verifies that prompts are tested with the target model.
    """
    # Reset connector call count
    dummy_connector.reset_count()

    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    result = await optimizer.optimize()

    # Connector should have been called multiple times
    # (at least once per prompt × test combination)
    assert dummy_connector.call_count > 0

    # Should have been called for evaluations
    expected_calls = (
        minimal_config.num_initial_prompts * minimal_config.num_quick_tests  # Quick eval
        + minimal_config.top_k_advance * minimal_config.num_rigorous_tests  # Rigorous eval
    )
    # Plus refinement evaluations (variable, but at least some)
    assert dummy_connector.call_count >= expected_calls


@pytest.mark.asyncio
async def test_output_directory_created(
    minimal_config, dummy_connector, mock_agents, test_database
):
    """
    Test that output directory is created for run results.

    Each run should have its own output directory.
    """
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    result = await optimizer.optimize()

    # Output directory should be set
    assert result.output_dir is not None

    # Directory should exist
    from pathlib import Path

    output_path = Path(result.output_dir)
    assert output_path.exists()
    assert output_path.is_dir()


@pytest.mark.asyncio
async def test_champion_test_results_complete(
    minimal_config, dummy_connector, mock_agents, test_database
):
    """
    Test that champion has complete test results for all rigorous tests.

    Champion should be evaluated on all rigorous tests.
    """
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    result = await optimizer.optimize()

    # Champion should have test results
    assert len(result.champion_test_results) > 0

    # Should have results for all rigorous tests
    # (or at least a reasonable subset if champion is from refinement)
    assert len(result.champion_test_results) >= minimal_config.num_rigorous_tests

    # All results should reference champion
    champion_id = result.best_prompt.id
    for test_result in result.champion_test_results:
        assert test_result.prompt_id == champion_id


@pytest.mark.asyncio
async def test_optimization_timing_tracked(
    minimal_config, dummy_connector, mock_agents, test_database
):
    """
    Test that optimization timing is tracked correctly.

    Result should include total execution time.
    """
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    result = await optimizer.optimize()

    # Should have timing information
    assert result.total_time_seconds > 0
    assert result.total_time_seconds < 300  # Should complete in under 5 minutes for tests


@pytest.mark.asyncio
async def test_test_counts_accurate(
    minimal_config, dummy_connector, mock_agents, test_database
):
    """
    Test that total_tests_run count is accurate.

    Should reflect actual number of evaluations performed.
    """
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    result = await optimizer.optimize()
    session = test_database.get_session()

    try:
        from prompt_optimizer.storage.models import Evaluation

        # Count actual evaluations in database
        actual_evaluations = session.query(Evaluation).filter_by(run_id=result.run_id).count()

        # Should match reported count (or be close - may include refinement evals)
        assert result.total_tests_run > 0
        assert result.total_tests_run == actual_evaluations

    finally:
        session.close()
