"""Generic runner for prompt optimization.

This module provides a reusable runner that accepts a connector and configuration
to run the optimization pipeline.
"""

import logging
from datetime import datetime
from pathlib import Path

from prompt_optimizer.config import OptimizerConfig
from prompt_optimizer.connectors import BaseConnector
from prompt_optimizer.optimizer import PromptOptimizer
from prompt_optimizer.reports import display_results
from prompt_optimizer.types import OptimizationResult

logger = logging.getLogger(__name__)


class OptimizationRunner:
    """Runner for executing prompt optimization with reporting."""

    def __init__(
        self,
        connector: BaseConnector,
        config: OptimizerConfig,
        verbose: bool = True,
    ):
        """Initialize the optimization runner.

        Args:
            connector: Connector for testing the target model
            config: Optimizer configuration with task specification
            verbose: Whether to print progress messages
        """
        self.connector = connector
        self.config = config
        self.results_root = config.results_path
        self.results_root.mkdir(parents=True, exist_ok=True)
        self.last_run_dir: Path | None = None
        self.verbose = verbose

        if config.task_spec is None:
            raise ValueError("OptimizerConfig must have task_spec populated")

        # Optimizer will be created in run() method with output_dir
        self.optimizer = None

    async def run(self) -> OptimizationResult:
        """Run the optimization pipeline with reporting.

        Returns:
            OptimizationResult with champion prompt and metrics
        """
        if self.verbose:
            self._print_header()

        # Create optimizer (it will create the output directory with proper run_id)
        self.optimizer = PromptOptimizer(
            model_client=self.connector,
            config=self.config,
        )

        result = await self.optimizer.optimize()

        # Get output directory from result
        run_output_dir = result.output_dir
        self.last_run_dir = Path(run_output_dir) if run_output_dir else None

        if self.verbose:
            display_results(result)

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

    def _prepare_run_directory(self, run_id: int | None) -> Path:
        """Create and return run-specific output directory."""
        if run_id is not None:
            folder_name = f"run-{run_id:04d}"
        else:
            timestamp = datetime.now().strftime("run-%Y%m%d-%H%M%S")
            folder_name = timestamp

        run_path = self.results_root / folder_name
        run_path.mkdir(parents=True, exist_ok=True)
        return run_path
