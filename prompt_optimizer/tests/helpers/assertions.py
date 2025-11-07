"""Custom assertions for prompt optimizer pipeline tests."""

from sqlalchemy.orm import Session

from prompt_optimizer.schemas import OptimizationResult
from prompt_optimizer.storage.models import Prompt


def assert_prompts_in_stage(session: Session, run_id: int, stage: str, expected_count: int) -> None:
    """
    Verify that the correct number of prompts exist in a specific stage.

    Args:
        session: Database session
        run_id: Run ID to check
        stage: Stage name (e.g., "initial", "quick_filter", "rigorous", "refined")
        expected_count: Expected number of prompts in this stage

    Raises:
        AssertionError: If count doesn't match
    """
    actual_prompts = session.query(Prompt).filter_by(run_id=run_id, stage=stage).all()
    actual_count = len(actual_prompts)

    assert (
        actual_count == expected_count
    ), f"Expected {expected_count} prompts in stage '{stage}', but found {actual_count}"


def assert_all_prompts_have_scores(
    session: Session, run_id: int, stage: str, score_field: str
) -> None:
    """
    Verify that all prompts in a stage have been evaluated (have scores).

    Args:
        session: Database session
        run_id: Run ID to check
        stage: Stage name to check prompts from
        score_field: Score field to check (e.g., "quick_score", "rigorous_score")

    Raises:
        AssertionError: If any prompt lacks a score
    """
    prompts = session.query(Prompt).filter_by(run_id=run_id, stage=stage).all()

    assert len(prompts) > 0, f"No prompts found in stage '{stage}' for run {run_id}"

    prompts_without_scores = [p for p in prompts if getattr(p, score_field) is None]

    assert (
        len(prompts_without_scores) == 0
    ), f"Found {len(prompts_without_scores)} prompts without {score_field} in stage '{stage}'"


def assert_top_k_selected(
    session: Session, run_id: int, k: int, source_stage: str, score_field: str
) -> None:
    """
    Verify that exactly top K prompts were selected based on scores.

    Args:
        session: Database session
        run_id: Run ID to check
        k: Number of top prompts expected
        source_stage: Stage to check prompts from
        score_field: Score field used for selection

    Raises:
        AssertionError: If selection is incorrect
    """
    prompts = (
        session.query(Prompt)
        .filter_by(run_id=run_id, stage=source_stage)
        .filter(getattr(Prompt, score_field).isnot(None))
        .all()
    )

    assert len(prompts) == k, f"Expected {k} prompts in stage '{source_stage}', found {len(prompts)}"

    # Verify they are sorted by score (top K should have highest scores)
    scores = [getattr(p, score_field) for p in prompts]
    assert scores == sorted(
        scores, reverse=True
    ), f"Top {k} prompts in '{source_stage}' are not sorted by {score_field}"


def assert_refinement_tracks_exist(session: Session, run_id: int, num_tracks: int) -> None:
    """
    Verify that refinement tracks were created correctly.

    Args:
        session: Database session
        run_id: Run ID to check
        num_tracks: Expected number of refinement tracks

    Raises:
        AssertionError: If tracks are not properly set up
    """
    refined_prompts = session.query(Prompt).filter_by(run_id=run_id, stage="refined").all()

    # Check that we have refined prompts
    assert (
        len(refined_prompts) > 0
    ), f"No refined prompts found for run {run_id} (expected {num_tracks} tracks)"

    # Check track IDs
    track_ids = set(p.track_id for p in refined_prompts if p.track_id is not None)
    assert (
        len(track_ids) == num_tracks
    ), f"Expected {num_tracks} unique track IDs, found {len(track_ids)}: {track_ids}"

    # Verify track IDs are in expected range [0, num_tracks-1]
    expected_track_ids = set(range(num_tracks))
    assert (
        track_ids == expected_track_ids
    ), f"Track IDs {track_ids} don't match expected {expected_track_ids}"


def assert_champion_is_best(result: OptimizationResult) -> None:
    """
    Verify that the champion prompt has the highest final score.

    Args:
        result: OptimizationResult from pipeline

    Raises:
        AssertionError: If champion is not the best performer
    """
    champion = result.best_prompt
    champion_score = champion.rigorous_score

    assert champion_score is not None, "Champion prompt has no rigorous_score"

    # Check that champion score is highest among all refined prompts
    all_final_scores = []
    for track in result.all_tracks:
        if track.final_prompt.rigorous_score is not None:
            all_final_scores.append(track.final_prompt.rigorous_score)

    assert (
        len(all_final_scores) > 0
    ), "No refined prompts have scores in optimization result"

    max_score = max(all_final_scores)
    assert (
        champion_score == max_score
    ), f"Champion score {champion_score} is not the highest score {max_score}"
