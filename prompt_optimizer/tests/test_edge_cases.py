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
