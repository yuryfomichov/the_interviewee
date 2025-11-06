"""Save all rigorous test questions and answers for the original system prompt."""

from pathlib import Path

import aiofiles

from prompt_optimizer.schemas import OptimizationResult


async def save_original_prompt_rigorous_results(
    result: OptimizationResult, output_dir: str
) -> Path | None:
    """
    Save all rigorous test questions and answers for the original system prompt.

    Args:
        result: Optimization result containing original prompt test results
        output_dir: Directory to save the file

    Returns:
        Path to saved file, or None if no original prompt
    """
    if not result.original_system_prompt or not result.original_system_prompt_test_results:
        return None

    qa_file = Path(output_dir) / "original_prompt_rigorous_results.txt"
    qa_file.parent.mkdir(parents=True, exist_ok=True)

    # Create a mapping of test_case_id to test case for easy lookup
    test_case_map = {test.id: test for test in result.rigorous_tests}

    lines = []
    lines.append("ORIGINAL SYSTEM PROMPT - RIGOROUS TEST RESULTS\n")
    lines.append("=" * 70 + "\n")
    lines.append(f"Prompt ID: {result.original_system_prompt.id}\n")
    lines.append(f"Overall Score: {result.original_system_prompt_rigorous_score:.2f}/10\n")
    lines.append(f"Total Tests: {len(result.original_system_prompt_test_results)}\n")
    lines.append("=" * 70 + "\n\n")

    # Group by category
    by_category: dict[str, list[tuple]] = {}
    for test_result in result.original_system_prompt_test_results:
        test_case = test_case_map.get(test_result.test_case_id)
        if test_case:
            category: str = test_case.category  # type: ignore[assignment]
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
            if test_case is None:
                continue
            lines.append(f"Test ID: {test_case.id}\n")
            lines.append(f"{'-' * 70}\n")
            lines.append(f"QUESTION:\n{test_case.input_message}\n\n")
            lines.append(f"EXPECTED BEHAVIOR:\n{test_case.expected_behavior}\n\n")
            lines.append(f"ANSWER:\n{test_result.model_response}\n\n")
            lines.append("EVALUATION:\n")
            lines.append(f"  Overall Score: {test_result.evaluation.overall:.2f}/10\n")
            lines.append(f"  Functionality: {test_result.evaluation.functionality}/10\n")
            lines.append(f"  Safety: {test_result.evaluation.safety}/10\n")
            lines.append(f"  Consistency: {test_result.evaluation.consistency}/10\n")
            lines.append(f"  Edge Case Handling: {test_result.evaluation.edge_case_handling}/10\n")
            lines.append(f"  Reasoning: {test_result.evaluation.reasoning}\n")
            lines.append("\n")

    async with aiofiles.open(qa_file, "w") as f:
        await f.write("".join(lines))

    print(f"Original prompt rigorous test results saved to: {qa_file}")
    return qa_file
