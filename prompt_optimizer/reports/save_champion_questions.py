"""Save all test questions used to evaluate the champion prompt."""

from pathlib import Path

import aiofiles

from prompt_optimizer.schemas import OptimizationResult


async def save_champion_questions(result: OptimizationResult, output_dir: str) -> Path:
    """
    Save all test questions used to evaluate the champion prompt.

    Args:
        result: Optimization result containing test cases
        output_dir: Directory to save the questions file

    Returns:
        Path to saved questions file
    """
    questions_file = Path(output_dir) / "champion_test_questions.md"
    questions_file.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("CHAMPION PROMPT TEST QUESTIONS\n")
    lines.append("=" * 70 + "\n")
    lines.append(f"Champion Prompt ID: {result.best_prompt.id}\n")
    lines.append(f"Total Test Questions: {len(result.rigorous_tests)}\n")
    lines.append("=" * 70 + "\n\n")

    # Group by category
    by_category: dict[str, list] = {}
    for test_case in result.rigorous_tests:
        category = test_case.category
        if category not in by_category:
            by_category[category] = []
        by_category[category].append(test_case)

    # Write questions by category
    categories = ["core", "edge", "boundary", "adversarial", "consistency", "format"]
    for cat in categories:
        cat_key: str = cat  # type: ignore[assignment]
        if cat_key not in by_category:
            continue

        lines.append(f"\n{'=' * 70}\n")
        lines.append(f"{cat_key.upper()} QUESTIONS ({len(by_category[cat_key])} tests)\n")
        lines.append(f"{'=' * 70}\n\n")

        for idx, test_case in enumerate(by_category[cat_key], 1):
            lines.append(f"{idx}. Test ID: {test_case.id}\n")
            lines.append(f"   Question: {test_case.input_message}\n")
            lines.append(f"   Expected: {test_case.expected_behavior}\n")
            lines.append("\n")

    async with aiofiles.open(questions_file, "w") as f:
        await f.write("".join(lines))

    print(f"Champion test questions saved to: {questions_file}")
    return questions_file
