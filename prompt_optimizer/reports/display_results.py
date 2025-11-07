"""Display optimization results to console."""

from prompt_optimizer.schemas import OptimizationResult


def display_results(result: OptimizationResult) -> None:
    """
    Display optimization results to console.

    Args:
        result: Optimization result containing champion prompt and metrics
    """
    print("\n" + "=" * 70)
    print("OPTIMIZATION COMPLETE!")
    print("=" * 70)
    print(f"\nChampion Prompt Score: {result.best_prompt.average_score:.2f}")
    print(f"Champion Track: {result.best_prompt.track_id}")
    print(f"\nTotal Tests Run: {result.total_tests_run}")
    print(f"Total Time: {result.total_time_seconds:.1f} seconds")
    print("\nTrack Comparison:")
    for track in result.all_tracks:
        # Find the best score achieved in this track (not the final iteration)
        best_score = max(track.score_progression) if track.score_progression else track.final_prompt.average_score
        initial_score = track.initial_prompt.average_score
        improvement = best_score - initial_score

        print(
            f"  Track {track.track_id}: "
            f"{initial_score:.2f} → {best_score:.2f} "
            f"({improvement:+.2f})"
        )
