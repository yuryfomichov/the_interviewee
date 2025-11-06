"""Save report for original system prompt after rigorous filter stage."""

from pathlib import Path


def save_original_prompt_rigorous_report(
    original_prompt,
    rigorous_tests: list,
    top_k_prompts: list,
    top_m_prompts: list,
    storage,
    output_dir: str,
) -> Path | None:
    """
    Save report for original system prompt after rigorous filter stage.

    Args:
        original_prompt: The original system prompt candidate
        rigorous_tests: Rigorous test cases used
        top_k_prompts: All prompts that underwent rigorous testing
        top_m_prompts: Prompts that advanced to meta-prompting
        storage: Storage instance to retrieve test results
        output_dir: Directory to save the report

    Returns:
        Path to saved report file, or None if no original prompt
    """
    if not original_prompt:
        return None

    report_file = Path(output_dir) / "original_prompt_rigorous_report.txt"
    report_file.parent.mkdir(parents=True, exist_ok=True)

    # Get test results for original prompt
    test_results = storage.get_prompt_evaluations(original_prompt.id)

    # Create test case mapping
    test_case_map = {test.id: test for test in rigorous_tests}

    # Determine ranking among top_k prompts
    sorted_prompts = sorted(top_k_prompts, key=lambda p: p.average_score or 0, reverse=True)
    rank = next(
        (i + 1 for i, p in enumerate(sorted_prompts) if p.id == original_prompt.id), None
    )

    # Check if advanced to meta-prompting
    advanced = original_prompt in top_m_prompts

    with report_file.open("w") as f:
        f.write("ORIGINAL SYSTEM PROMPT - RIGOROUS TEST REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write("SUMMARY\n")
        f.write("-" * 70 + "\n")
        f.write(f"Score: {original_prompt.average_score:.2f}/10\n")
        f.write(f"Rank: {rank}/{len(top_k_prompts)} among prompts in rigorous testing\n")
        f.write(f"Status: {'✓ ADVANCED to meta-prompting' if advanced else '✗ FILTERED OUT'}\n")
        f.write(f"Total Rigorous Tests: {len(test_results)}\n\n")

        # Performance breakdown
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

        # Detailed Q&A
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

        for category in ["core", "edge", "boundary", "adversarial", "consistency", "format"]:
            if category not in by_category:
                continue

            f.write(f"\n{category.upper()} TESTS\n")
            f.write("-" * 70 + "\n\n")

            for test_case, test_result in by_category[category]:
                f.write(f"Test ID: {test_case.id}\n")
                f.write(f"Question: {test_case.input_message}\n")
                f.write(f"Expected: {test_case.expected_behavior}\n\n")
                f.write(f"Answer:\n{test_result.model_response}\n\n")
                f.write(f"Evaluation:\n")
                f.write(f"  Overall Score: {test_result.evaluation.overall:.2f}/10\n")
                f.write(f"  Functionality: {test_result.evaluation.functionality}/10\n")
                f.write(f"  Safety: {test_result.evaluation.safety}/10\n")
                f.write(f"  Consistency: {test_result.evaluation.consistency}/10\n")
                f.write(f"  Edge Case: {test_result.evaluation.edge_case_handling}/10\n")
                f.write(f"  Reasoning: {test_result.evaluation.reasoning}\n")
                f.write(f"\n")

        # Original prompt text
        f.write("=" * 70 + "\n")
        f.write("ORIGINAL PROMPT TEXT\n")
        f.write("=" * 70 + "\n")
        f.write(original_prompt.prompt_text)
        f.write("\n")

    print(f"Original prompt rigorous test report saved to: {report_file}")
    return report_file
