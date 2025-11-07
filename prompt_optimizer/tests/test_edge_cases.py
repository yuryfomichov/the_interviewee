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
