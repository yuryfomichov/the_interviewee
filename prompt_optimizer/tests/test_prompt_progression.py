"""Test prompt progression through pipeline stages."""

import pytest

from prompt_optimizer.optimizer.orchestrator import PromptOptimizer


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "selected_attr,source_attr,count_attr,score_field,description",
    [
        ("top_k_prompts", "initial_prompts", "top_k_advance", "quick_score", "top-K from initial prompts"),
        ("top_m_prompts", "top_k_prompts", "top_m_refine", "rigorous_score", "top-M from top-K prompts"),
    ],
)
async def test_selection_by_score(
    minimal_config,
    dummy_connector,
    mock_agents,
    test_database,
    selected_attr,
    source_attr,
    count_attr,
    score_field,
    description,
):
    """
    Test that prompt selection correctly picks top N prompts by score.

    Parameterized to test both top-K and top-M selection logic.
    """
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    result = await optimizer.optimize()

    # Get selected and source prompts
    selected_prompts = getattr(result, selected_attr)
    source_prompts = getattr(result, source_attr)
    expected_count = getattr(minimal_config, count_attr)

    # Verify correct number selected
    assert len(selected_prompts) == expected_count, f"Expected {expected_count} {description}"

    # Verify selected prompts have the highest scores from source
    selected_scores = sorted(
        [getattr(p, score_field) for p in selected_prompts if getattr(p, score_field) is not None],
        reverse=True,
    )
    all_scores = sorted(
        [getattr(p, score_field) for p in source_prompts if getattr(p, score_field) is not None],
        reverse=True,
    )

    # Top N scores should match top N scores from all source prompts
    top_n_scores = all_scores[:expected_count]
    assert selected_scores == top_n_scores, f"Selected {description} don't have highest {score_field} values"


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
