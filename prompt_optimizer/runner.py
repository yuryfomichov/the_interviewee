"""Generic runner for prompt optimization.

This module provides a reusable runner that accepts a connector and configuration
to run the optimization pipeline.
"""

import logging
from pathlib import Path

from prompt_optimizer.config import OptimizerConfig
from prompt_optimizer.connectors import BaseConnector
from prompt_optimizer.optimizer import PromptOptimizer
from prompt_optimizer.reporter import (
    display_results,
    save_champion_prompt,
    save_optimization_report,
)
from prompt_optimizer.types import OptimizationResult

logger = logging.getLogger(__name__)


class OptimizationRunner:
    """Runner for executing prompt optimization with reporting."""

    def __init__(
        self,
        connector: BaseConnector,
        config: OptimizerConfig,
        output_dir: str | Path = "prompt_optimizer/data",
        verbose: bool = True,
    ):
        """Initialize the optimization runner.

        Args:
            connector: Connector for testing the target model
            config: Optimizer configuration with task specification
            output_dir: Directory for saving results
            verbose: Whether to print progress messages
        """
        self.connector = connector
        self.config = config
        self.output_dir = Path(output_dir)
        self.verbose = verbose

        if config.task_spec is None:
            raise ValueError("OptimizerConfig must have task_spec populated")

        self.optimizer = PromptOptimizer(
            model_client=connector,
            config=config,
        )

    async def run(self) -> OptimizationResult:
        """Run the optimization pipeline with reporting.

        Returns:
            OptimizationResult with champion prompt and metrics
        """
        if self.verbose:
            self._print_header()

        result = await self.optimizer.optimize()

        if self.verbose:
            display_results(result)

        save_champion_prompt(result, output_dir=str(self.output_dir))
        save_optimization_report(result, self.config.task_spec, output_dir=str(self.output_dir))

        return result

    def _print_header(self) -> None:
        """Print optimization header."""
        print("=" * 70)
        print("PROMPT OPTIMIZATION PIPELINE")
        print("=" * 70)
        print()
        print(f"Task: {self.config.task_spec.task_description}")
        print()
        print("Configuration:")
        print(f"  Initial prompts: {self.config.num_initial_prompts}")
        print(f"  Quick tests: {self.config.num_quick_tests}")
        print(f"  Rigorous tests: {self.config.num_rigorous_tests}")
        print("  Models:")
        print(f"    Generator: {self.config.generator_llm.model}")
        print(f"    Test designer: {self.config.test_designer_llm.model}")
        print(f"    Evaluator: {self.config.evaluator_llm.model}")
        print(f"    Refiner: {self.config.refiner_llm.model}")
        if self.config.parallel_execution:
            print(
                f"  Parallel execution: enabled "
                f"(max concurrent evaluations: {self.config.max_concurrent_evaluations})"
            )
        else:
            print("  Parallel execution: disabled")
        print()
        print("Starting optimization...")
        print()
