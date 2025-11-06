"""Save all questions and answers for the champion prompt."""

from pathlib import Path

from prompt_optimizer.types import OptimizationResult


def save_champion_qa_results(result: OptimizationResult, output_dir: str) -> Path:
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
