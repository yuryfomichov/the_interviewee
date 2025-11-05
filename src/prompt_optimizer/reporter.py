"""Results reporting for prompt optimization.

This module handles displaying optimization results to the console
and saving detailed reports to files.
"""

from pathlib import Path

from prompt_optimizer.types import OptimizationResult, TaskSpec


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


def save_champion_prompt(
    result: OptimizationResult, output_dir: str = "prompt_optimizer/data"
) -> Path:
    """
    Save champion prompt to file.

    Args:
        result: Optimization result containing champion prompt
        output_dir: Directory to save the champion prompt

    Returns:
        Path to saved champion prompt file
    """
    output_file = Path(output_dir) / "champion_prompt.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text(result.best_prompt.prompt_text)
    print(f"\nChampion prompt saved to: {output_file}")

    # Display the champion prompt
    print("\n" + "=" * 70)
    print("CHAMPION SYSTEM PROMPT:")
    print("=" * 70)
    print(result.best_prompt.prompt_text)
    print("=" * 70)

    return output_file


def save_optimization_report(
    result: OptimizationResult,
    task_spec: TaskSpec,
    output_dir: str = "prompt_optimizer/data",
) -> Path:
    """
    Save detailed optimization report to file.

    Args:
        result: Optimization result
        task_spec: Task specification used for optimization
        output_dir: Directory to save the report

    Returns:
        Path to saved report file
    """
    report_file = Path(output_dir) / "optimization_report.txt"
    report_file.parent.mkdir(parents=True, exist_ok=True)

    with report_file.open("w") as f:
        f.write("PROMPT OPTIMIZATION REPORT\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Task: {task_spec.task_description}\n\n")
        f.write(f"Champion Score: {result.best_prompt.average_score:.2f}\n")
        f.write(f"Total Tests: {result.total_tests_run}\n")
        f.write(f"Total Time: {result.total_time_seconds:.1f}s\n\n")
        f.write("Track Results:\n")
        for track in result.all_tracks:
            f.write(f"\nTrack {track.track_id}:\n")
            f.write(f"  Initial: {track.initial_prompt.average_score:.2f}\n")
            f.write(f"  Final: {track.final_prompt.average_score:.2f}\n")
            f.write(f"  Improvement: +{track.improvement:.2f}\n")
            f.write(f"  Iterations: {len(track.iterations)}\n")
            f.write(
                f"  Score progression: {', '.join(f'{s:.2f}' for s in track.score_progression)}\n"
            )
        f.write("\n" + "=" * 70 + "\n")
        f.write("CHAMPION PROMPT:\n")
        f.write("=" * 70 + "\n")
        f.write(result.best_prompt.prompt_text)

    print(f"\nDetailed report saved to: {report_file}")
    return report_file
