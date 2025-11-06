"""Save all rigorous test questions and answers for the original system prompt."""

from pathlib import Path

from prompt_optimizer.types import OptimizationResult


def save_original_prompt_rigorous_results(
    result: OptimizationResult = None,
    output_dir: str = None,
    original_prompt=None,
    rigorous_tests: list = None,
    top_k_prompts: list = None,
    top_m_prompts: list = None,
    storage=None,
) -> Path | None:
    """
    Save all rigorous test questions and answers for the original system prompt.

    Can be called in two ways:
    1. From runner.py at end: pass result and output_dir
    2. From selection stage: pass individual parameters

    Args:
        result: Optimization result (for end-of-run call)
        output_dir: Directory to save the file
        original_prompt: The original system prompt candidate (for stage call)
        rigorous_tests: Rigorous test cases used (for stage call)
        top_k_prompts: All prompts in rigorous testing (for stage call)
        top_m_prompts: Prompts that advanced to meta-prompting (for stage call)
        storage: Storage instance to retrieve test results (for stage call)

    Returns:
        Path to saved file, or None if no original prompt
    """
    # Handle two calling patterns
    if result is not None:
        # Called from runner.py at end with OptimizationResult
        if not result.original_system_prompt or not result.original_system_prompt_test_results:
            return None
        original_prompt = result.original_system_prompt
        test_results = result.original_system_prompt_test_results
        test_case_map = {test.id: test for test in result.rigorous_tests}
        overall_score = result.original_system_prompt_rigorous_score
        rank = None
        advanced = None
    else:
        # Called from selection stage with individual parameters
        if not original_prompt:
            return None
        test_results = storage.get_prompt_evaluations(original_prompt.id)
        test_case_map = {test.id: test for test in rigorous_tests}
        overall_score = original_prompt.average_score

        # Calculate rank among top_k prompts
        sorted_prompts = sorted(top_k_prompts, key=lambda p: p.average_score or 0, reverse=True)
        rank = next(
            (i + 1 for i, p in enumerate(sorted_prompts) if p.id == original_prompt.id), None
        )

        # Check if advanced to meta-prompting
        advanced = original_prompt in top_m_prompts

    qa_file = Path(output_dir) / "original_prompt_rigorous_results.txt"
    qa_file.parent.mkdir(parents=True, exist_ok=True)

    with qa_file.open("w") as f:
        f.write("ORIGINAL SYSTEM PROMPT - RIGOROUS TEST RESULTS\n")
        f.write("=" * 70 + "\n\n")

        f.write("SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Score: {overall_score:.2f}/10\n")
        if rank is not None:
            f.write(f"Rank: {rank}/{len(top_k_prompts)} among prompts in rigorous testing\n")
        if advanced is not None:
            f.write(f"Status: {'✓ ADVANCED to meta-prompting' if advanced else '✗ FILTERED OUT'}\n")
        f.write(f"Total Tests: {len(test_results)}\n")
        f.write("\n")

        # Performance breakdown by category
        if test_results:
            scores_by_category = {}
            for test_result in test_results:
                test_case = test_case_map.get(test_result.test_case_id)
                if test_case:
                    category = test_case.category
                    if category not in scores_by_category:
                        scores_by_category[category] = []
                    scores_by_category[category].append(test_result.evaluation.overall)

            f.write("PERFORMANCE BY CATEGORY\n")
            f.write("-" * 70 + "\n")
            for category in ["core", "edge", "boundary", "adversarial", "consistency", "format"]:
                if category in scores_by_category:
                    scores = scores_by_category[category]
                    avg = sum(scores) / len(scores)
                    f.write(f"{category.upper():15} {avg:.2f}/10 ({len(scores)} tests)\n")
            f.write("\n")

        f.write("=" * 70 + "\n")
        f.write("DETAILED TEST RESULTS\n")
        f.write("=" * 70 + "\n\n")

        # Group by category
        by_category = {}
        for test_result in test_results:
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

        # Original prompt text
        f.write("=" * 70 + "\n")
        f.write("ORIGINAL PROMPT TEXT\n")
        f.write("=" * 70 + "\n")
        f.write(original_prompt.prompt_text)
        f.write("\n")

    print(f"Original prompt rigorous test results saved to: {qa_file}")
    return qa_file
