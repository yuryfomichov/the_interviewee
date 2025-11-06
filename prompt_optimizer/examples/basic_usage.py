"""Example: Basic usage of the prompt optimizer with a custom connector.

This example shows how to create a custom connector and run optimization.
"""

import asyncio
import os

from prompt_optimizer import BaseConnector, OptimizerConfig, OptimizationRunner


class MyCustomConnector(BaseConnector):
    """Example custom connector implementation."""

    def test_prompt(self, system_prompt: str, message: str) -> str:
        """Test your custom model with the given prompt and message.

        Replace this implementation with your actual model/system.
        """
        # This is just a placeholder - replace with your actual model call
        return f"Response to: {message}"


async def main():
    """Run optimization with a custom connector."""
    # 1. Create your custom connector
    connector = MyCustomConnector()

    # 2. Configure the optimizer
    config = OptimizerConfig(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        task_description="Answer technical questions clearly and accurately",
        success_criteria=[
            "Provides accurate technical information",
            "Explains concepts clearly",
            "Gives practical examples",
        ],
        num_initial_prompts=5,  # Reduced for faster testing
        num_quick_tests=3,
        num_rigorous_tests=10,
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
    print(f"\nBest prompt score: {result.best_prompt.average_score:.2f}")
    print(f"Champion prompt:\n{result.best_prompt.prompt_text}")


if __name__ == "__main__":
    asyncio.run(main())
