"""Main runner for prompt optimization.

This module orchestrates the complete optimization pipeline,
coordinating the RAG engine, optimizer configuration, and results reporting.
"""

import logging
import os

from prompt_optimizer.model_clients import FunctionModelClient
from prompt_optimizer.optimizer import PromptOptimizer
from prompt_optimizer.types import OptimizationResult

from .config import create_optimizer_config
from .connector import create_rag_test_function
from .reporter import display_results, save_champion_prompt, save_optimization_report

logger = logging.getLogger(__name__)


async def run_optimization() -> OptimizationResult | None:
    """
    Run the complete prompt optimization pipeline.

    This function:
    1. Validates environment setup (API keys)
    2. Initializes RAG engine test function
    3. Configures optimizer
    4. Runs optimization pipeline
    5. Displays and saves results

    Returns:
        OptimizationResult if successful, None if setup validation fails
    """
    print("=" * 70)
    print("AI INTERVIEWEE SYSTEM PROMPT OPTIMIZER")
    print("=" * 70)
    print()

    # Check for OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("❌ ERROR: OPENAI_API_KEY not found!")
        print()
        print("Please set your OpenAI API key in one of these ways:")
        print("  1. Add to .env file: OPENAI_API_KEY=your_key_here")
        print("  2. Set environment variable: export OPENAI_API_KEY=your_key_here")
        print()
        return None

    # Initialize RAG engine test function
    print("Initializing RAG engine...")
    test_function = create_rag_test_function()

    # Create optimizer configuration with API key
    optimizer_config = create_optimizer_config(api_key=api_key)

    print("Configuration:")
    print(f"  Initial prompts: {optimizer_config.num_initial_prompts}")
    print(f"  Quick tests: {optimizer_config.num_quick_tests}")
    print(f"  Rigorous tests: {optimizer_config.num_rigorous_tests}")
    print(f"  Model: {optimizer_config.generator_llm.model}")
    print()

    # Create task specification from config
    task_spec = optimizer_config.task_spec
    if task_spec is None:
        raise ValueError("task_spec must be provided on the optimizer configuration.")

    # Initialize optimizer
    model_client = FunctionModelClient(test_function)
    optimizer = PromptOptimizer(
        model_client=model_client,
        config=optimizer_config,
    )

    # Run optimization
    print("\nStarting optimization pipeline...")
    print("This will take 15-20 minutes with GPT-4o")
    print()

    try:
        result = await optimizer.optimize()

        # Display results
        display_results(result)

        # Save outputs
        save_champion_prompt(result)
        save_optimization_report(result, task_spec)

        return result

    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        raise
