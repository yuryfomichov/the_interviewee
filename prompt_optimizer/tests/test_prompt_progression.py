"""Test prompt progression through pipeline stages."""

import pytest

from prompt_optimizer.optimizer.orchestrator import PromptOptimizer


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

    # Verify champion has rigorous_score
    champion = result.best_prompt
    champion_score = champion.rigorous_score
    assert champion_score is not None, "Champion prompt has no rigorous_score"

    # Check that champion score is highest among all refined prompts
    all_final_scores = []
    for track in result.all_tracks:
        if track.final_prompt.rigorous_score is not None:
            all_final_scores.append(track.final_prompt.rigorous_score)

    assert len(all_final_scores) > 0, "No refined prompts have scores in optimization result"

    max_score = max(all_final_scores)
    assert champion_score == max_score, f"Champion score {champion_score} is not the highest score {max_score}"


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
