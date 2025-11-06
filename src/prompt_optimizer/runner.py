"""RAG-specific runner for prompt optimization.

This module provides the integration between the RAG engine
and the generic prompt optimization runner.
"""

import logging
import os

from prompt_optimizer.runner import OptimizationRunner
from prompt_optimizer.types import OptimizationResult

from .config import create_optimizer_config
from .connector import RAGConnector

logger = logging.getLogger(__name__)


async def run_optimization() -> OptimizationResult | None:
    """
    Run the RAG engine prompt optimization pipeline.

    This function:
    1. Validates environment setup (API keys)
    2. Initializes RAG engine connector
    3. Configures optimizer
    4. Runs optimization pipeline with reporting

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

    # Initialize RAG engine connector
    print("Initializing RAG engine...")
    rag_connector = RAGConnector()

    # Create optimizer configuration with API key
    optimizer_config = create_optimizer_config(api_key=api_key)

    print(f"This will take 15-20 minutes with {optimizer_config.generator_llm.model}")
    print()

    try:
        # Run optimization using the generic runner
        runner = OptimizationRunner(
            connector=rag_connector,
            config=optimizer_config,
            verbose=True,
        )
        return await runner.run()

    except Exception as e:
        logger.error(f"Optimization failed: {e}", exc_info=True)
        raise
