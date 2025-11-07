"""Test prompt progression through pipeline stages."""

import pytest

from prompt_optimizer.optimizer.orchestrator import PromptOptimizer
from prompt_optimizer.tests.helpers import (
    assert_all_prompts_have_scores,
    assert_champion_is_best,
    assert_prompts_in_stage,
    assert_top_k_selected,
)


@pytest.mark.asyncio
async def test_prompts_advance_through_all_stages(
    minimal_config, dummy_connector, mock_agents, test_database
):
    """
    Test that prompts correctly advance through all pipeline stages.

    Verifies the complete progression:
    initial -> quick_filter -> rigorous -> refined
    """
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    result = await optimizer.optimize()

    # Use result object which contains snapshots at each stage
    # Stage 1: All initial prompts created
    assert len(result.initial_prompts) == minimal_config.num_initial_prompts

    # Stage 2: Top K advanced to quick_filter
    assert len(result.top_k_prompts) == minimal_config.top_k_advance

    # Stage 3: Top M advanced to rigorous
    assert len(result.top_m_prompts) == minimal_config.top_m_refine

    # Stage 4: Refined prompts exist (in refinement tracks)
    assert len(result.all_tracks) > 0
    for track in result.all_tracks:
        assert len(track.iterations) > 0


@pytest.mark.asyncio
async def test_all_initial_prompts_evaluated(
    minimal_config, dummy_connector, mock_agents, test_database
):
    """
    Test that all initial prompts receive quick evaluation scores.

    Every prompt in the initial stage should be evaluated and have a quick_score.
    """
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    result = await optimizer.optimize()

    # All initial prompts should have quick_score after evaluation
    assert len(result.initial_prompts) == minimal_config.num_initial_prompts

    for prompt in result.initial_prompts:
        assert (
            prompt.quick_score is not None
        ), f"Prompt {prompt.id} has no quick_score"
        assert 0 <= prompt.quick_score <= 10, f"Invalid quick_score: {prompt.quick_score}"


@pytest.mark.asyncio
async def test_top_k_selection_by_score(
    minimal_config, dummy_connector, mock_agents, test_database
):
    """
    Test that exactly top K prompts are selected based on quick_score.

    The selected prompts should have the highest quick_scores.
    """
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    result = await optimizer.optimize()

    # Verify correct number selected
    assert len(result.top_k_prompts) == minimal_config.top_k_advance

    # Verify that top K prompts have the highest scores from initial prompts
    quick_scores = sorted([p.quick_score for p in result.top_k_prompts], reverse=True)
    all_scores = sorted([p.quick_score for p in result.initial_prompts if p.quick_score is not None], reverse=True)

    # Top K scores should match top K scores from all prompts
    top_k_scores = all_scores[: minimal_config.top_k_advance]
    assert quick_scores == top_k_scores


@pytest.mark.asyncio
async def test_top_m_selection_by_rigorous_score(
    minimal_config, dummy_connector, mock_agents, test_database
):
    """
    Test that exactly top M prompts are selected based on rigorous_score.

    The selected prompts should have the highest rigorous_scores.
    """
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    result = await optimizer.optimize()

    # Verify correct number selected
    assert len(result.top_m_prompts) == minimal_config.top_m_refine

    # Verify top M prompts have the highest rigorous_scores from top K
    rigorous_scores = sorted([p.rigorous_score for p in result.top_m_prompts if p.rigorous_score is not None], reverse=True)
    all_scores = sorted([p.rigorous_score for p in result.top_k_prompts if p.rigorous_score is not None], reverse=True)

    top_m_scores = all_scores[: minimal_config.top_m_refine]
    assert rigorous_scores == top_m_scores


@pytest.mark.asyncio
async def test_champion_has_highest_score(
    minimal_config, dummy_connector, mock_agents, test_database
):
    """
    Test that the champion prompt has the highest final score.

    The champion (best_prompt) should be the best performer from all refinement tracks.
    """
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    result = await optimizer.optimize()

    # Use custom assertion
    assert_champion_is_best(result)


@pytest.mark.asyncio
async def test_scores_populated_at_each_stage(
    minimal_config, dummy_connector, mock_agents, test_database
):
    """
    Test that scores are properly populated at each evaluation stage.

    - Initial prompts should have quick_score
    - Quick_filter prompts should have both quick_score and rigorous_score
    - Rigorous prompts should have rigorous_score
    - Refined prompts should have rigorous_score
    """
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    result = await optimizer.optimize()
    run_id = result.run_id
    session = test_database.get_session()

    try:
        from prompt_optimizer.storage.models import Prompt

        # Initial prompts: should have quick_score
        initial_prompts = session.query(Prompt).filter_by(run_id=run_id, stage="initial").all()
        for prompt in initial_prompts:
            assert prompt.quick_score is not None

        # Quick_filter prompts: should have both quick_score and rigorous_score
        quick_prompts = session.query(Prompt).filter_by(run_id=run_id, stage="quick_filter").all()
        for prompt in quick_prompts:
            assert prompt.quick_score is not None
            assert prompt.rigorous_score is not None

        # Rigorous prompts: should have rigorous_score
        rigorous_prompts = session.query(Prompt).filter_by(run_id=run_id, stage="rigorous").all()
        for prompt in rigorous_prompts:
            assert prompt.rigorous_score is not None

        # Refined prompts: should have rigorous_score (from refinement evaluation)
        refined_prompts = session.query(Prompt).filter_by(run_id=run_id, stage="refined").all()
        for prompt in refined_prompts:
            if prompt.iteration > 0:  # Refinement iterations should have scores
                assert prompt.rigorous_score is not None

    finally:
        session.close()


@pytest.mark.asyncio
async def test_original_prompt_tracked_separately(
    config_with_original_prompt, dummy_connector, mock_agents, test_database
):
    """
    Test that original prompt is tracked separately through the pipeline.

    The original prompt should:
    - Be marked with is_original_system_prompt=True
    - Have strategy="original_system_prompt"
    - Be evaluated but tracked independently
    """
    optimizer = PromptOptimizer(
        model_client=dummy_connector,
        config=config_with_original_prompt,
        database=test_database,
    )

    result = await optimizer.optimize()
    run_id = result.run_id
    session = test_database.get_session()

    try:
        from prompt_optimizer.storage.models import Prompt

        # Find original prompt in database
        original_prompt = (
            session.query(Prompt)
            .filter_by(run_id=run_id, is_original_system_prompt=True)
            .first()
        )

        assert original_prompt is not None
        assert original_prompt.strategy == "original_system_prompt"
        assert original_prompt.prompt_text == config_with_original_prompt.task_spec.current_prompt

        # Original prompt should be evaluated on quick tests
        assert original_prompt.quick_score is not None

        # Original prompt should be evaluated on rigorous tests for comparison
        assert original_prompt.rigorous_score is not None

    finally:
        session.close()


@pytest.mark.asyncio
async def test_prompt_counts_consistent_across_stages(
    realistic_config, dummy_connector, mock_agents, test_database
):
    """
    Test that prompt counts are consistent with configuration at each stage.

    Uses realistic config with larger numbers to verify scaling.
    """
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=realistic_config, database=test_database
    )

    result = await optimizer.optimize()

    # Verify counts in result match configuration
    assert len(result.initial_prompts) == realistic_config.num_initial_prompts
    assert len(result.top_k_prompts) == realistic_config.top_k_advance
    assert len(result.top_m_prompts) == realistic_config.top_m_refine
    assert len(result.all_tracks) == realistic_config.top_m_refine

    # Verify database state matches
    run_id = result.run_id
    session = test_database.get_session()

    try:
        assert_prompts_in_stage(
            session, run_id, "initial", realistic_config.num_initial_prompts
        )
        assert_prompts_in_stage(
            session, run_id, "quick_filter", realistic_config.top_k_advance
        )
        assert_prompts_in_stage(session, run_id, "rigorous", realistic_config.top_m_refine)

    finally:
        session.close()
