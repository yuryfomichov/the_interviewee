"""Save detailed pipeline report showing prompt progression through all stages."""

from pathlib import Path

import aiofiles

from prompt_optimizer.schemas import OptimizationResult


async def save_pipeline_report(result: OptimizationResult, output_dir: str) -> Path:
    """
    Save detailed pipeline report showing all prompts and their progression.

    Args:
        result: Optimization result with all prompts and test data
        output_dir: Directory to save the report

    Returns:
        Path to saved report file
    """
    report_file = Path(output_dir) / "pipeline_report.md"
    report_file.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append("=" * 80 + "\n")
    lines.append("PIPELINE REPORT - Prompt Progression Through All Stages\n")
    lines.append("=" * 80 + "\n\n")

    # STAGE 1: Initial Prompts
    lines.append("STAGE 1: Initial Prompts Generated\n")
    lines.append("-" * 80 + "\n")
    for prompt in result.initial_prompts:
        marker = " (original system prompt)" if prompt.is_original_system_prompt else ""
        lines.append(f"  - {prompt.id}{marker}\n")
    lines.append(f"\nTotal: {len(result.initial_prompts)} prompts generated\n\n")

    # STAGE 3: Quick Filter Evaluation
    lines.append("STAGE 3: Quick Filter Evaluation\n")
    lines.append("-" * 80 + "\n")

    # Sort by quick_score descending for better readability
    sorted_initial = sorted(
        result.initial_prompts,
        key=lambda p: p.quick_score or 0,
        reverse=True
    )

    top_k_ids = {p.id for p in result.top_k_prompts}

    for prompt in sorted_initial:
        if prompt.quick_score is not None:
            promoted = "→ PROMOTED to rigorous" if prompt.id in top_k_ids else "→ FILTERED OUT"
            marker = " (comparison only)" if prompt.is_original_system_prompt and prompt.id in top_k_ids else ""
            lines.append(
                f"  {prompt.id}: {prompt.quick_score:.2f}/10 {promoted}{marker}\n"
            )
        else:
            lines.append(f"  {prompt.id}: Not evaluated\n")

    lines.append(f"\nTop {len(result.top_k_prompts)} selected for rigorous testing\n\n")

    # STAGE 6: Rigorous Evaluation
    lines.append("STAGE 6: Rigorous Evaluation\n")
    lines.append("-" * 80 + "\n")

    # Sort by rigorous_score descending
    sorted_rigorous = sorted(
        result.top_k_prompts,
        key=lambda p: p.rigorous_score or 0,
        reverse=True
    )

    top_m_ids = {p.id for p in result.top_m_prompts}

    for prompt in sorted_rigorous:
        if prompt.rigorous_score is not None:
            if prompt.id in top_m_ids:
                # Find which track this prompt started
                track_num = None
                for track in result.all_tracks:
                    if track.initial_prompt.id == prompt.id:
                        track_num = track.track_id
                        break
                track_info = f" (Track {track_num})" if track_num is not None else ""
                promoted = f"→ PROMOTED to refinement{track_info}"
            else:
                marker = " (comparison only)" if prompt.is_original_system_prompt else ""
                promoted = f"→ FILTERED OUT{marker}"

            lines.append(
                f"  {prompt.id}: {prompt.rigorous_score:.2f}/10 {promoted}\n"
            )
        else:
            lines.append(f"  {prompt.id}: Not evaluated\n")

    lines.append(f"\nTop {len(result.top_m_prompts)} selected for refinement\n\n")

    # STAGE 8: Refinement Tracks
    if result.all_tracks:
        lines.append("STAGE 8: Refinement Tracks\n")
        lines.append("-" * 80 + "\n\n")

        for track in result.all_tracks:
            lines.append(f"Track {track.track_id} (starting from {track.initial_prompt.id}):\n")

            for iteration_prompt in track.iterations:
                score = iteration_prompt.rigorous_score or 0
                stage_marker = f" [{iteration_prompt.stage}]"
                best_marker = " ← BEST" if score == max(track.score_progression) else ""

                lines.append(
                    f"  Iteration {iteration_prompt.iteration}: "
                    f"{iteration_prompt.id} ({score:.2f}/10)"
                    f"{stage_marker}{best_marker}\n"
                )

            lines.append(
                f"  Score progression: {' → '.join(f'{s:.2f}' for s in track.score_progression)}\n"
            )
            lines.append(f"  Improvement: {track.improvement:+.2f}\n\n")

    # FINAL RESULT
    lines.append("=" * 80 + "\n")
    lines.append("FINAL CHAMPION\n")
    lines.append("=" * 80 + "\n")
    lines.append(f"Champion Prompt ID: {result.best_prompt.id}\n")
    lines.append(f"Score: {result.best_prompt.rigorous_score:.2f}/10\n")
    lines.append(f"Track: {result.best_prompt.track_id}\n")
    lines.append(f"Iteration: {result.best_prompt.iteration}\n")
    lines.append(f"Stage: {result.best_prompt.stage}\n")

    async with aiofiles.open(report_file, "w") as f:
        await f.write("".join(lines))

    print(f"Pipeline report saved to: {report_file}")
    return report_file
