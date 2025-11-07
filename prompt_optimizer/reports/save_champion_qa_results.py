"""Save all questions and answers for the champion prompt."""

from pathlib import Path

import aiofiles

from prompt_optimizer.schemas import OptimizationResult


async def save_champion_qa_results(result: OptimizationResult, output_dir: str) -> Path:
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

    # Build content as string first
    lines = []
    lines.append("CHAMPION PROMPT Q&A RESULTS\n")
    lines.append("=" * 70 + "\n")
    lines.append(f"Champion Prompt ID: {result.best_prompt.id}\n")
    lines.append(f"Overall Score: {result.best_prompt.average_score:.2f}\n")
    lines.append(f"Total Tests: {len(result.champion_test_results)}\n")
    lines.append("=" * 70 + "\n\n")

    # Weaknesses summary
    failures = [test for test in result.champion_test_results if test.evaluation.overall < 7.0]
    if failures:
        lines.append("WEAKNESSES SUMMARY\n")
        lines.append("-" * 70 + "\n")
        lines.append(
            f"Found {len(failures)} test(s) with scores below 7.0 "
            f"(out of {len(result.champion_test_results)} total):\n\n"
        )
        for test_result in failures:
            test_case = test_case_map.get(test_result.test_case_id)
            if test_case:
                lines.append(
                    f"  • [{test_case.category.upper()}] Score: {test_result.evaluation.overall:.2f} "
                    f"- {test_case.input_message[:60]}...\n"
                )
        lines.append("\n")
    else:
        lines.append("WEAKNESSES SUMMARY\n")
        lines.append("-" * 70 + "\n")
        lines.append("No significant weaknesses - all tests scored 7.0 or above! ✓\n\n")

    # Group by category
    by_category: dict[str, list[tuple]] = {}
    for test_result in result.champion_test_results:
        test_case = test_case_map.get(test_result.test_case_id)
        if test_case:
            category = test_case.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append((test_case, test_result))

    # Write results by category
    categories = ["core", "edge", "boundary", "adversarial", "consistency", "format"]
    for cat in categories:
        cat_key: str = cat  # type: ignore[assignment]
        if cat_key not in by_category:
            continue

        lines.append(f"\n{'=' * 70}\n")
        lines.append(f"{cat_key.upper()} TESTS\n")
        lines.append(f"{'=' * 70}\n\n")

        for test_case, test_result in by_category[cat_key]:
            lines.append(f"Test ID: {test_case.id}\n")
            lines.append(f"{'-' * 70}\n")
            lines.append(f"QUESTION:\n{test_case.input_message}\n\n")
            lines.append(f"EXPECTED BEHAVIOR:\n{test_case.expected_behavior}\n\n")
            lines.append(f"ANSWER:\n{test_result.model_response}\n\n")
            lines.append(f"EVALUATION:\n")
            lines.append(f"  Overall Score: {test_result.evaluation.overall:.2f}/10\n")
            lines.append(f"  Functionality: {test_result.evaluation.functionality}/10\n")
            lines.append(f"  Safety: {test_result.evaluation.safety}/10\n")
            lines.append(f"  Consistency: {test_result.evaluation.consistency}/10\n")
            lines.append(f"  Edge Case Handling: {test_result.evaluation.edge_case_handling}/10\n")
            lines.append(f"  Reasoning: {test_result.evaluation.reasoning}\n")
            lines.append(f"\n")

    async with aiofiles.open(qa_file, "w") as f:
        await f.write("".join(lines))

    print(f"Champion Q&A results saved to: {qa_file}")
    return qa_file
