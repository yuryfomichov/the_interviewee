"""Save tested prompts with scores to JSON file."""

import json
from pathlib import Path

import aiofiles

from prompt_optimizer.types import OptimizationResult


async def save_prompts_json(result: OptimizationResult, output_dir: str) -> Path:
    """
    Save all tested prompts with their quick evaluation scores to JSON file.

    Args:
        result: Optimization result containing all prompts
        output_dir: Directory to save the JSON file

    Returns:
        Path to saved JSON file
    """
    prompts_file = Path(output_dir) / "prompts_with_scores.json"
    prompts_file.parent.mkdir(parents=True, exist_ok=True)

    # Prepare prompts data with quick scores
    prompts_data = {
        "initial_prompts": [
            {
                "id": prompt.id,
                "track_id": prompt.track_id,
                "prompt_text": prompt.prompt_text,
                "average_score": prompt.average_score,
                "is_original_system_prompt": prompt.is_original_system_prompt,
            }
            for prompt in result.initial_prompts
        ],
        "quick_filter_top_prompts": [
            {
                "id": prompt.id,
                "track_id": prompt.track_id,
                "prompt_text": prompt.prompt_text,
                "average_score": prompt.average_score,
                "is_original_system_prompt": prompt.is_original_system_prompt,
            }
            for prompt in result.top_k_prompts
        ],
        "rigorous_filter_top_prompts": [
            {
                "id": prompt.id,
                "track_id": prompt.track_id,
                "prompt_text": prompt.prompt_text,
                "average_score": prompt.average_score,
                "is_original_system_prompt": prompt.is_original_system_prompt,
            }
            for prompt in result.top_m_prompts
        ],
        "champion": {
            "id": result.best_prompt.id,
            "track_id": result.best_prompt.track_id,
            "prompt_text": result.best_prompt.prompt_text,
            "average_score": result.best_prompt.average_score,
            "is_original_system_prompt": result.best_prompt.is_original_system_prompt,
        },
        "summary": {
            "total_initial_prompts": len(result.initial_prompts),
            "quick_filter_top_count": len(result.top_k_prompts),
            "rigorous_filter_top_count": len(result.top_m_prompts),
            "champion_score": result.best_prompt.average_score,
        },
    }

    # Add original prompt info if available
    if result.original_system_prompt:
        prompts_data["original_system_prompt"] = {
            "id": result.original_system_prompt.id,
            "track_id": result.original_system_prompt.track_id,
            "prompt_text": result.original_system_prompt.prompt_text,
            "quick_score": result.original_system_prompt.average_score,
            "rigorous_score": result.original_system_prompt_rigorous_score,
        }

    # Write to JSON file with pretty formatting
    async with aiofiles.open(prompts_file, "w") as f:
        await f.write(json.dumps(prompts_data, indent=2, ensure_ascii=False))

    print(f"Prompts with scores saved to: {prompts_file}")
    return prompts_file
