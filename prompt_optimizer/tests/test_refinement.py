"""Test refinement stage behavior and iteration logic."""

import pytest

from prompt_optimizer.optimizer.orchestrator import PromptOptimizer


@pytest.mark.asyncio
async def test_refinement_structure(
    minimal_config, dummy_connector, mock_agents, test_database
):
    """
    Test that refinement creates proper track structure with isolated tracks.

    Verifies:
    - Creates expected number of parallel tracks
    - Each track starts from a different top-M prompt
    - Score progression is tracked for each track
    """
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    result = await optimizer.optimize()

    # Should have top_m_refine tracks
    assert len(result.all_tracks) == minimal_config.top_m_refine

    # Each track should have unique track_id (0, 1, 2, ...)
    track_ids = [track.track_id for track in result.all_tracks]
    assert set(track_ids) == set(range(minimal_config.top_m_refine))

    # All tracks should start from different prompts (isolation)
    initial_prompt_ids = {track.initial_prompt.id for track in result.all_tracks}
    assert len(initial_prompt_ids) == minimal_config.top_m_refine

    # Score progression should be tracked for each track
    for track in result.all_tracks:
        assert isinstance(track.score_progression, list)
        assert len(track.score_progression) > 0

        # First score should match initial prompt score
        assert track.score_progression[0] == track.initial_prompt.rigorous_score

        # All scores should be valid
        for score in track.score_progression:
            assert 0 <= score <= 10


@pytest.mark.asyncio
async def test_refinement_keeps_best_score(
    minimal_config, dummy_connector, mock_agents, test_database
):
    """
    Test that refinement keeps the best-performing prompt in each track.

    This is the core refinement behavior: the final prompt should have
    the highest score from all iterations in that track.
    """
    optimizer = PromptOptimizer(
        model_client=dummy_connector, config=minimal_config, database=test_database
    )

    result = await optimizer.optimize()

    for track in result.all_tracks:
        final_score = track.final_prompt.rigorous_score

        # Final score should be the best score from the track's progression
        max_score_in_track = max(track.score_progression)
        assert final_score == max_score_in_track
