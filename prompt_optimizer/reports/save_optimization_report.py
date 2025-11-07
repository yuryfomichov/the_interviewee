"""Save detailed optimization report to file."""

from pathlib import Path

import aiofiles

from prompt_optimizer.schemas import OptimizationResult, TaskSpec


async def save_optimization_report(
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

    lines = []
    lines.append("PROMPT OPTIMIZATION REPORT\n")
    lines.append("=" * 70 + "\n\n")
    lines.append(f"Task: {task_spec.task_description}\n\n")
    lines.append(f"Champion Score: {result.best_prompt.average_score:.2f}\n")
    lines.append(f"Total Tests: {result.total_tests_run}\n")
    lines.append(f"Total Time: {result.total_time_seconds:.1f}s\n\n")

    # Original system prompt performance
    if result.original_system_prompt and result.original_system_prompt_rigorous_score:
        lines.append("\n" + "=" * 70 + "\n")
        lines.append("ORIGINAL SYSTEM PROMPT PERFORMANCE (RIGOROUS TESTS)\n")
        lines.append("=" * 70 + "\n")
        lines.append(
            f"Rigorous Test Score: {result.original_system_prompt_rigorous_score:.2f}/10\n"
        )
        lines.append(
            f"Status: {'Advanced to refinement' if result.original_system_prompt in result.top_m_prompts else 'Filtered out after quick tests'}\n"
        )
        if (
            result.original_system_prompt_rigorous_score is not None
            and result.best_prompt.average_score is not None
        ):
            improvement = (
                result.best_prompt.average_score - result.original_system_prompt_rigorous_score
            )
            improvement_pct = improvement / result.original_system_prompt_rigorous_score * 100
        else:
            improvement = 0.0
            improvement_pct = 0.0
        lines.append(f"Improvement over original: {improvement:+.2f} ({improvement_pct:+.1f}%)\n")
        lines.append(
            f"\nNote: Both scores based on {len(result.rigorous_tests)} rigorous tests for fair comparison.\n"
        )

    lines.append("\n" + "=" * 70 + "\n")
    lines.append("TRACK RESULTS\n")
    lines.append("=" * 70 + "\n")
    for track in result.all_tracks:
        lines.append(f"\nTrack {track.track_id}:\n")
        lines.append(f"  Initial: {track.initial_prompt.average_score:.2f}\n")
        lines.append(f"  Final: {track.final_prompt.average_score:.2f}\n")
        lines.append(f"  Improvement: {track.improvement:+.2f}\n")
        lines.append(f"  Iterations: {len(track.iterations)}\n")
        lines.append(
            f"  Score progression: {', '.join(f'{s:.2f}' for s in track.score_progression)}\n"
        )

        # Weaknesses identified during refinement
        if track.weaknesses_history:
            lines.append("\n  Weaknesses Identified:\n")
            for weakness in track.weaknesses_history:
                lines.append(f"    Iteration {weakness.iteration}:\n")
                lines.append(f"      {weakness.description}\n")
                if weakness.failed_test_descriptions:
                    lines.append("      Failed tests:\n")
                    for test_desc in weakness.failed_test_descriptions[:3]:
                        lines.append(f"        - {test_desc}\n")

    lines.append("\n" + "=" * 70 + "\n")
    lines.append("CHAMPION PROMPT:\n")
    lines.append("=" * 70 + "\n")
    lines.append(result.best_prompt.prompt_text)

    async with aiofiles.open(report_file, "w") as f:
        await f.write("".join(lines))

    print(f"\nDetailed report saved to: {report_file}")
    return report_file
