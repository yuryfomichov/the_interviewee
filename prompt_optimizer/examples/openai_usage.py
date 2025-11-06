"""Example: Using the built-in OpenAI connector.

This example shows how to optimize prompts for an OpenAI model.
"""

import asyncio
import os

from prompt_optimizer import OpenAIConnector, OptimizerConfig, OptimizationRunner


async def main():
    """Run optimization with OpenAI connector."""
    # Check for API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        return

    # 1. Create OpenAI connector
    connector = OpenAIConnector(
        api_key=api_key,
        model="gpt-4o-mini",  # or "gpt-4", "gpt-3.5-turbo", etc.
    )

    # 2. Configure the optimizer
    config = OptimizerConfig(
        openai_api_key=api_key,  # Used by the optimizer agents
        task_description="Answer customer support questions professionally",
        success_criteria=[
            "Responds politely and professionally",
            "Addresses the customer's concern directly",
            "Provides actionable solutions",
            "Maintains a helpful tone",
        ],
        num_initial_prompts=10,
        num_quick_tests=5,
        num_rigorous_tests=20,
    )

    # 3. Create and run the optimization runner
    runner = OptimizationRunner(
        connector=connector,
        config=config,
        output_dir="optimization_results",
        verbose=True,
    )

    # 4. Run optimization
    result = await runner.run()

    # 5. Access the results
    print(f"\nOptimization complete!")
    print(f"Best prompt score: {result.best_prompt.average_score:.2f}")
    print(f"Total tests run: {result.total_tests_run}")
    print(f"Total time: {result.total_time_seconds:.1f} seconds")


if __name__ == "__main__":
    asyncio.run(main())
