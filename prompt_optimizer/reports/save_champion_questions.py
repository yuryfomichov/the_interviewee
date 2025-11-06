"""Save all test questions used to evaluate the champion prompt."""

from pathlib import Path

from prompt_optimizer.types import OptimizationResult


def save_champion_questions(result: OptimizationResult, output_dir: str) -> Path:
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
