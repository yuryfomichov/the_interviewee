"""Save report for original system prompt after quick filter stage."""

from pathlib import Path

import aiofiles

from prompt_optimizer.storage import EvaluationConverter


async def save_original_prompt_quick_report(
    original_prompt,
    quick_tests: list,
    initial_prompts: list,
    top_k_prompts: list,
    context,
    output_dir: str,
) -> Path | None:
    """
    Save report for original system prompt after quick filter stage.

    Args:
        original_prompt: The original system prompt candidate
        quick_tests: Quick test cases used
        initial_prompts: All initial prompts for ranking comparison
        top_k_prompts: Prompts that advanced to rigorous testing
        context: Run context for database access
        output_dir: Directory to save the report

    Returns:
        Path to saved report file, or None if no original prompt
    """
    if not original_prompt:
        return None

    report_file = Path(output_dir) / "original_prompt_quick_report.md"
    report_file.parent.mkdir(parents=True, exist_ok=True)

    # Get test results for original prompt from database
    db_evaluations = context.eval_repo.get_by_prompt(original_prompt.id)
    test_results = [EvaluationConverter.from_db(ev) for ev in db_evaluations]

    # Create test case mapping
    test_case_map = {test.id: test for test in quick_tests}

    # Filter test results to only include quick tests
    quick_test_results = [tr for tr in test_results if tr.test_case_id in test_case_map]

    # Calculate score from the quick test results
    # If no quick test evaluations found, fall back to the prompt's quick_score field
    if quick_test_results:
        quick_score = sum(tr.evaluation.overall for tr in quick_test_results) / len(quick_test_results)
    elif original_prompt.quick_score is not None:
        quick_score = original_prompt.quick_score
    else:
        quick_score = 0.0

    # Determine ranking based on quick_score (not average_score which includes rigorous tests)
    sorted_prompts = sorted(initial_prompts, key=lambda p: p.quick_score or 0, reverse=True)
    rank = next((i + 1 for i, p in enumerate(sorted_prompts) if p.id == original_prompt.id), None)

    # Check if advanced (compare by ID, not object identity)
    top_k_ids = {p.id for p in top_k_prompts}
    advanced = original_prompt.id in top_k_ids

    lines = []
    lines.append("ORIGINAL SYSTEM PROMPT - QUICK TEST REPORT\n")
    lines.append("=" * 70 + "\n\n")

    lines.append("SUMMARY\n")
    lines.append("-" * 70 + "\n")
    lines.append(f"Score: {quick_score:.2f}/10\n")
    lines.append(f"Rank: {rank}/{len(initial_prompts)} among initial prompts\n")
    lines.append(f"Status: {'✓ ADVANCED to rigorous testing' if advanced else '✗ FILTERED OUT'}\n")
    lines.append(f"Total Quick Tests: {len(quick_test_results)}\n\n")

    # Performance breakdown
    if quick_test_results:
        scores_by_category: dict[str, list[float]] = {}
        for test_result in quick_test_results:
            test_case = test_case_map.get(test_result.test_case_id)
            if test_case:
                category = test_case.category
                if category not in scores_by_category:
                    scores_by_category[category] = []
                scores_by_category[category].append(test_result.evaluation.overall)

        lines.append("PERFORMANCE BY CATEGORY\n")
        lines.append("-" * 70 + "\n")
        for category in ["core", "edge", "boundary", "adversarial", "consistency", "format"]:
            if category in scores_by_category:
                scores = scores_by_category[category]
                avg = sum(scores) / len(scores)
                lines.append(f"{category.upper():15} {avg:.2f}/10 ({len(scores)} tests)\n")
        lines.append("\n")

    # Detailed Q&A
    lines.append("=" * 70 + "\n")
    lines.append("DETAILED TEST RESULTS\n")
    lines.append("=" * 70 + "\n\n")

    # Group by category
    by_category: dict[str, list[tuple]] = {}
    for test_result in quick_test_results:
        test_case = test_case_map.get(test_result.test_case_id)
        if test_case:
            category = test_case.category
            if category not in by_category:
                by_category[category] = []
            by_category[category].append((test_case, test_result))

    for category in ["core", "edge", "boundary", "adversarial", "consistency", "format"]:
        if category not in by_category:
            continue

        lines.append(f"\n{category.upper()} TESTS\n")
        lines.append("-" * 70 + "\n\n")

        for test_case, test_result in by_category[category]:
            lines.append(f"Test ID: {test_case.id}\n")
            lines.append(f"Question: {test_case.input_message}\n")
            lines.append(f"Expected: {test_case.expected_behavior}\n\n")
            lines.append(f"Answer:\n{test_result.model_response}\n\n")
            lines.append("Evaluation:\n")
            lines.append(f"  Overall Score: {test_result.evaluation.overall:.2f}/10\n")
            lines.append(f"  Functionality: {test_result.evaluation.functionality}/10\n")
            lines.append(f"  Safety: {test_result.evaluation.safety}/10\n")
            lines.append(f"  Consistency: {test_result.evaluation.consistency}/10\n")
            lines.append(f"  Edge Case: {test_result.evaluation.edge_case_handling}/10\n")
            lines.append(f"  Reasoning: {test_result.evaluation.reasoning}\n")
            lines.append("\n")

    # Original prompt text
    lines.append("=" * 70 + "\n")
    lines.append("ORIGINAL PROMPT TEXT\n")
    lines.append("=" * 70 + "\n")
    lines.append(original_prompt.prompt_text)
    lines.append("\n")

    async with aiofiles.open(report_file, "w") as f:
        await f.write("".join(lines))

    print(f"Original prompt quick test report saved to: {report_file}")
    return report_file
