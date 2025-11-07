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
    report_file = Path(output_dir) / "optimization_report.md"
    report_file.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("PROMPT OPTIMIZATION REPORT\n")
    lines.append("=" * 70 + "\n\n")
    lines.append(f"Task: {task_spec.task_description}\n\n")
    lines.append(f"Champion Score: {result.best_prompt.rigorous_score:.2f}\n")
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
            and result.best_prompt.rigorous_score is not None
        ):
            improvement = (
                result.best_prompt.rigorous_score - result.original_system_prompt_rigorous_score
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
        # Calculate best score achieved (not just the final iteration)
        best_score = max(track.score_progression) if track.score_progression else track.final_prompt.rigorous_score
        best_improvement = best_score - track.initial_prompt.rigorous_score
        best_iter = track.score_progression.index(best_score) if track.score_progression else 0

        lines.append(f"\nTrack {track.track_id}:\n")
        lines.append(f"  Initial: {track.initial_prompt.rigorous_score:.2f}\n")
        lines.append(f"  Best: {best_score:.2f} (iteration {best_iter})\n")
        lines.append(f"  Final: {track.final_prompt.rigorous_score:.2f}\n")
        lines.append(f"  Best improvement: {best_improvement:+.2f}\n")
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
    lines.append("\n\n")

    # Champion prompt weaknesses (current weaknesses based on test results)
    lines.append("=" * 70 + "\n")
    lines.append("CHAMPION PROMPT WEAKNESSES (Current Issues)\n")
    lines.append("=" * 70 + "\n")
    champion_failures = [
        test for test in result.champion_test_results if test.evaluation.overall < 7.0
    ]
    if champion_failures:
        lines.append(
            f"\nFound {len(champion_failures)} test(s) with scores below 7.0 "
            f"(out of {len(result.champion_test_results)} total tests):\n\n"
        )
        for i, test_result in enumerate(champion_failures, 1):
            # Find the corresponding test case
            test_case = next(
                (t for t in result.rigorous_tests if t.id == test_result.test_case_id), None
            )
            if test_case:
                lines.append(f"{i}. Test: {test_case.input_message[:80]}...\n")
                lines.append(f"   Expected: {test_case.expected_behavior[:60]}...\n")
                lines.append(f"   Score: {test_result.evaluation.overall:.2f}/10\n")
                lines.append(f"   Issue: {test_result.evaluation.reasoning[:100]}...\n\n")
            else:
                # Test case not found - display what we have
                lines.append(f"{i}. Test ID: {test_result.test_case_id}\n")
                lines.append(f"   Score: {test_result.evaluation.overall:.2f}/10\n")
                lines.append(f"   Issue: {test_result.evaluation.reasoning[:100]}...\n")
                lines.append(f"   (Test case details not found in rigorous tests)\n\n")
    else:
        lines.append("\nNo significant weaknesses found - all tests scored 7.0 or above! ✓\n")

    # Champion refinement history (weaknesses identified during development)
    champion_track = next(
        (t for t in result.all_tracks if t.final_prompt.id == result.best_prompt.id), None
    )
    if champion_track and champion_track.weaknesses_history:
        lines.append("\n" + "=" * 70 + "\n")
        lines.append("CHAMPION REFINEMENT HISTORY (Weaknesses Addressed)\n")
        lines.append("=" * 70 + "\n")
        lines.append(
            f"\nTrack {champion_track.track_id} refined through {len(champion_track.iterations)} iterations:\n\n"
        )
        for weakness in champion_track.weaknesses_history:
            lines.append(f"Iteration {weakness.iteration}:\n")
            lines.append(f"  Issue: {weakness.description}\n")
            if weakness.failed_test_descriptions:
                lines.append(f"  Failed tests: {len(weakness.failed_test_descriptions)}\n")
                for test_desc in weakness.failed_test_descriptions[:2]:
                    lines.append(f"    - {test_desc[:80]}...\n")
            lines.append("\n")

    # Original prompt weaknesses
    if result.original_system_prompt and result.original_system_prompt_test_results:
        lines.append("\n" + "=" * 70 + "\n")
        lines.append("ORIGINAL SYSTEM PROMPT WEAKNESSES\n")
        lines.append("=" * 70 + "\n")
        original_failures = [
            test for test in result.original_system_prompt_test_results if test.evaluation.overall < 7.0
        ]
        if original_failures:
            lines.append(
                f"\nFound {len(original_failures)} test(s) with scores below 7.0 "
                f"(out of {len(result.original_system_prompt_test_results)} total tests):\n\n"
            )
            for i, test_result in enumerate(original_failures, 1):
                test_case = next(
                    (t for t in result.rigorous_tests if t.id == test_result.test_case_id), None
                )
                if test_case:
                    lines.append(f"{i}. Test: {test_case.input_message[:80]}...\n")
                    lines.append(f"   Expected: {test_case.expected_behavior[:60]}...\n")
                    lines.append(f"   Score: {test_result.evaluation.overall:.2f}/10\n")
                    lines.append(f"   Issue: {test_result.evaluation.reasoning[:100]}...\n\n")
                else:
                    # Test case not found - display what we have
                    lines.append(f"{i}. Test ID: {test_result.test_case_id}\n")
                    lines.append(f"   Score: {test_result.evaluation.overall:.2f}/10\n")
                    lines.append(f"   Issue: {test_result.evaluation.reasoning[:100]}...\n")
                    lines.append(f"   (Test case details not found in rigorous tests)\n\n")
        else:
            lines.append("\nNo significant weaknesses found - all tests scored 7.0 or above! ✓\n")

    async with aiofiles.open(report_file, "w") as f:
        await f.write("".join(lines))

    print(f"\nDetailed report saved to: {report_file}")
    return report_file
