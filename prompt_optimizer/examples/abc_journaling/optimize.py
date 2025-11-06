"""Optimize the ABC journaling system prompt using the OpenAI connector.

Usage:
    export OPENAI_API_KEY="your-api-key-here"
    python -m prompt_optimizer.examples.abc_journaling.optimize
"""

from __future__ import annotations

import asyncio
import logging
import os

from dotenv import load_dotenv

from prompt_optimizer.connectors.openai_connector import OpenAIConnector
from prompt_optimizer.examples.abc_journaling.config import create_optimizer_config
from prompt_optimizer.runner import OptimizationRunner

# Load environment variables when executed as a module
load_dotenv()

# Configure logging similar to the main optimizer entry point
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


async def main() -> None:
    """Run the optimization pipeline for the ABC journaling assistant."""
    print("=" * 70)
    print("ABC JOURNALING PROMPT OPTIMIZER")
    print("=" * 70)
    print()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not found!")
        print()
        print("Please set your OpenAI API key in one of these ways:")
        print("  1. Add to .env file: OPENAI_API_KEY=your_key_here")
        print("  2. Set environment variable: export OPENAI_API_KEY=your_key_here")
        print()
        return

    # Initialize connector for the target model that will be optimized
    target_model = "gpt-5-nano"
    print(f"Initializing OpenAI connector with target model: {target_model}")
    connector = OpenAIConnector(api_key=api_key, model=target_model)

    # Build optimizer configuration scoped to the journaling task
    optimizer_config = create_optimizer_config(api_key=api_key)
    task_spec = optimizer_config.task_spec

    print(f"Task: {task_spec.task_description}")
    print(f"Current prompt length: {len(task_spec.current_prompt or '')} characters")
    print(f"Generator model: {optimizer_config.generator_llm.model}")
    print(f"Evaluator model: {optimizer_config.evaluator_llm.model}")
    print()

    output_dir = optimizer_config.results_path
    output_dir.mkdir(parents=True, exist_ok=True)

    runner = OptimizationRunner(
        connector=connector,
        config=optimizer_config,
        verbose=True,
    )

    result, last_run_dir = await runner.run()

    print()
    print("=" * 70)
    print("OPTIMIZATION COMPLETE")
    print("=" * 70)
    average_score = result.best_prompt.average_score
    if average_score is not None:
        print(f"\nChampion score: {average_score:.2f}")
    print(f"Champion prompt ID: {result.best_prompt.id}")
    if result.best_prompt.track_id is not None:
        print(f"Refinement track: {result.best_prompt.track_id}")
    results_dir = last_run_dir or output_dir
    print(f"\nResults saved in: {results_dir}")
    print()


if __name__ == "__main__":
    asyncio.run(main())
