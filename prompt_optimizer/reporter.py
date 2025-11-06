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

        # Original system prompt performance
        if result.original_system_prompt:
            f.write("\n" + "=" * 70 + "\n")
            f.write("ORIGINAL SYSTEM PROMPT PERFORMANCE\n")
            f.write("=" * 70 + "\n")
            f.write(f"Score: {result.original_system_prompt.average_score:.2f}\n")
            f.write(
                f"Status: {'Advanced to refinement' if result.original_system_prompt in result.stage2_top3 else 'Filtered out'}\n"
            )
            if result.original_system_prompt.average_score:
                improvement = (
                    result.best_prompt.average_score
                    - result.original_system_prompt.average_score
                )
                f.write(
                    f"Improvement over original: {improvement:+.2f} "
                    f"({improvement/result.original_system_prompt.average_score*100:+.1f}%)\n"
                )

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


def save_champion_qa_results(
    result: OptimizationResult, output_dir: str = "prompt_optimizer/data"
) -> Path:
    """
    Save all questions and answers for the champion prompt.

    Args:
        result: Optimization result containing champion test results
        output_dir: Directory to save the Q&A file

    Returns:
        Path to saved Q&A file
    """
    qa_file = Path(output_dir) / "champion_qa_results.txt"
    qa_file.parent.mkdir(parents=True, exist_ok=True)

    # Create a mapping of test_case_id to test case for easy lookup
    test_case_map = {test.id: test for test in result.rigorous_tests}

    with qa_file.open("w") as f:
        f.write("CHAMPION PROMPT Q&A RESULTS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Champion Prompt ID: {result.best_prompt.id}\n")
        f.write(f"Overall Score: {result.best_prompt.average_score:.2f}\n")
        f.write(f"Total Tests: {len(result.champion_test_results)}\n")
        f.write("=" * 70 + "\n\n")

        # Group by category
        by_category = {}
        for test_result in result.champion_test_results:
            test_case = test_case_map.get(test_result.test_case_id)
            if test_case:
                category = test_case.category
                if category not in by_category:
                    by_category[category] = []
                by_category[category].append((test_case, test_result))

        # Write results by category
        for category in ["core", "edge", "boundary", "adversarial", "consistency", "format"]:
            if category not in by_category:
                continue

            f.write(f"\n{'=' * 70}\n")
            f.write(f"{category.upper()} TESTS\n")
            f.write(f"{'=' * 70}\n\n")

            for test_case, test_result in by_category[category]:
                f.write(f"Test ID: {test_case.id}\n")
                f.write(f"{'-' * 70}\n")
                f.write(f"QUESTION:\n{test_case.input_message}\n\n")
                f.write(f"EXPECTED BEHAVIOR:\n{test_case.expected_behavior}\n\n")
                f.write(f"ANSWER:\n{test_result.model_response}\n\n")
                f.write(f"EVALUATION:\n")
                f.write(f"  Overall Score: {test_result.evaluation.overall:.2f}/10\n")
                f.write(f"  Functionality: {test_result.evaluation.functionality}/10\n")
                f.write(f"  Safety: {test_result.evaluation.safety}/10\n")
                f.write(f"  Consistency: {test_result.evaluation.consistency}/10\n")
                f.write(f"  Edge Case Handling: {test_result.evaluation.edge_case_handling}/10\n")
                f.write(f"  Reasoning: {test_result.evaluation.reasoning}\n")
                f.write(f"\n")

    print(f"Champion Q&A results saved to: {qa_file}")
    return qa_file


def save_champion_questions(
    result: OptimizationResult, output_dir: str = "prompt_optimizer/data"
) -> Path:
    """
    Save all test questions used to evaluate the champion prompt.

    Args:
        result: Optimization result containing test cases
        output_dir: Directory to save the questions file

    Returns:
        Path to saved questions file
    """
    questions_file = Path(output_dir) / "champion_test_questions.txt"
    questions_file.parent.mkdir(parents=True, exist_ok=True)

    with questions_file.open("w") as f:
        f.write("CHAMPION PROMPT TEST QUESTIONS\n")
        f.write("=" * 70 + "\n")
        f.write(f"Champion Prompt ID: {result.best_prompt.id}\n")
        f.write(f"Total Test Questions: {len(result.rigorous_tests)}\n")
        f.write("=" * 70 + "\n\n")

        # Group by category
        by_category = {}
        for test_case in result.rigorous_tests:
            category = test_case.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(test_case)

        # Write questions by category
        for category in ["core", "edge", "boundary", "adversarial", "consistency", "format"]:
            if category not in by_category:
                continue

            f.write(f"\n{'=' * 70}\n")
            f.write(f"{category.upper()} QUESTIONS ({len(by_category[category])} tests)\n")
            f.write(f"{'=' * 70}\n\n")

            for idx, test_case in enumerate(by_category[category], 1):
                f.write(f"{idx}. Test ID: {test_case.id}\n")
                f.write(f"   Question: {test_case.input_message}\n")
                f.write(f"   Expected: {test_case.expected_behavior}\n")
                f.write(f"\n")

    print(f"Champion test questions saved to: {questions_file}")
    return questions_file
