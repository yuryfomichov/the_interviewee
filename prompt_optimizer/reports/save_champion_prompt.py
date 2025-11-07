"""Save champion prompt to file."""

from pathlib import Path

import aiofiles

from prompt_optimizer.schemas import OptimizationResult


async def save_champion_prompt(result: OptimizationResult, output_dir: str) -> Path:
    """
    Save champion prompt to file.

    Args:
        result: Optimization result containing champion prompt
        output_dir: Directory to save the champion prompt

    Returns:
        Path to saved champion prompt file
    """
    output_file = Path(output_dir) / "champion_prompt.md"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    async with aiofiles.open(output_file, "w") as f:
        await f.write(result.best_prompt.prompt_text)

    print(f"\nChampion prompt saved to: {output_file}")

    # Display the champion prompt
    print("\n" + "=" * 70)
    print("CHAMPION SYSTEM PROMPT:")
    print("=" * 70)
    print(result.best_prompt.prompt_text)
    print("=" * 70)

    return output_file
