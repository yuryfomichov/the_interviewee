"""Save detailed optimization report to file."""

from pathlib import Path

from prompt_optimizer.types import OptimizationResult, TaskSpec


def save_optimization_report(
    result: OptimizationResult,
    task_spec: TaskSpec,
    output_dir: str,
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

        # Original system prompt performance
        if result.original_system_prompt and result.original_system_prompt_rigorous_score:
            f.write("\n" + "=" * 70 + "\n")
            f.write("ORIGINAL SYSTEM PROMPT PERFORMANCE (RIGOROUS TESTS)\n")
            f.write("=" * 70 + "\n")
            f.write(f"Rigorous Test Score: {result.original_system_prompt_rigorous_score:.2f}/10\n")
            f.write(
                f"Status: {'Advanced to refinement' if result.original_system_prompt in result.top_m_prompts else 'Filtered out after quick tests'}\n"
            )
            improvement = (
                result.best_prompt.average_score
                - result.original_system_prompt_rigorous_score
            )
            improvement_pct = (improvement / result.original_system_prompt_rigorous_score * 100)
            f.write(
                f"Improvement over original: {improvement:+.2f} "
                f"({improvement_pct:+.1f}%)\n"
            )
            f.write(f"\nNote: Both scores based on {len(result.rigorous_tests)} rigorous tests for fair comparison.\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("TRACK RESULTS\n")
        f.write("=" * 70 + "\n")
        for track in result.all_tracks:
            f.write(f"\nTrack {track.track_id}:\n")
            f.write(f"  Initial: {track.initial_prompt.average_score:.2f}\n")
            f.write(f"  Final: {track.final_prompt.average_score:.2f}\n")
            f.write(f"  Improvement: +{track.improvement:.2f}\n")
            f.write(f"  Iterations: {len(track.iterations)}\n")
            f.write(
                f"  Score progression: {', '.join(f'{s:.2f}' for s in track.score_progression)}\n"
            )

            # Weaknesses identified during refinement
            if track.weaknesses_history:
                f.write(f"\n  Weaknesses Identified:\n")
                for weakness in track.weaknesses_history:
                    f.write(f"    Iteration {weakness.iteration}:\n")
                    f.write(f"      {weakness.description}\n")
                    if weakness.failed_test_descriptions:
                        f.write(f"      Failed tests:\n")
                        for test_desc in weakness.failed_test_descriptions[:3]:
                            f.write(f"        - {test_desc}\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("CHAMPION PROMPT:\n")
        f.write("=" * 70 + "\n")
        f.write(result.best_prompt.prompt_text)

    print(f"\nDetailed report saved to: {report_file}")
    return report_file
