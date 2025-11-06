"""Display optimization results to console."""

from prompt_optimizer.types import OptimizationResult


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
        print(
            f"  Track {track.track_id}: "
            f"{track.initial_prompt.average_score:.2f} → "
            f"{track.final_prompt.average_score:.2f} "
            f"(+{track.improvement:.2f})"
        )
