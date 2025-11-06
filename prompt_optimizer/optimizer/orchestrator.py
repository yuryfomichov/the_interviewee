"""Main prompt optimizer orchestrator using stage pipeline."""

import logging
import os
import time

from prompt_optimizer.config import OptimizerConfig
from prompt_optimizer.optimizer.base_stage import BaseStage
from prompt_optimizer.optimizer.context import RunContext
from prompt_optimizer.optimizer.stages import (
    EvaluatePromptsStage,
    GeneratePromptsStage,
    GenerateTestsStage,
    RefinementStage,
    SelectTopPromptsStage,
)
from prompt_optimizer.storage import Storage
from prompt_optimizer.types import OptimizationResult

logger = logging.getLogger(__name__)


class PromptOptimizer:
    """Main orchestrator for the prompt optimization pipeline."""

    def __init__(
        self,
        model_client,
        config: OptimizerConfig | None = None,
        storage: Storage | None = None,
    ):
        """
        Initialize the optimizer.

        Args:
            model_client: Client for testing the target model.
            config: Optimizer configuration with a populated task_spec.
            storage: Storage instance (creates new if None).
        """
        self.model_client = model_client
        if config is None:
            raise ValueError("OptimizerConfig with task_spec must be provided.")
        self.config = config
        self.storage = storage or Storage(self.config.storage_path)

        # Set OpenAI API key in environment if provided in config
        if self.config.openai_api_key:
            os.environ["OPENAI_API_KEY"] = self.config.openai_api_key
            logger.info("OpenAI API key set from config")

        # Initialize stages pipeline
        self.stages: list[BaseStage] = self._create_stages()

    def _create_stages(self) -> list[BaseStage]:
        """
        Create the optimization pipeline stages.

        Returns:
            List of stage instances in execution order
        """
        stage_kwargs = {
            "config": self.config,
            "storage": self.storage,
            "model_client": self.model_client,
            "progress_callback": self._print_progress,
        }

        return [
            # Stage 1: Generate initial prompts
            GeneratePromptsStage(**stage_kwargs),
            # Stage 2: Generate quick tests
            GenerateTestsStage("quick", **stage_kwargs),
            # Stage 3: Evaluate with quick tests
            EvaluatePromptsStage("quick_filter", **stage_kwargs),
            # Stage 4: Select top K
            SelectTopPromptsStage(self.config.top_k_advance, "quick", **stage_kwargs),
            # Stage 5: Generate rigorous tests
            GenerateTestsStage("rigorous", **stage_kwargs),
            # Stage 6: Evaluate with rigorous tests
            EvaluatePromptsStage("rigorous", **stage_kwargs),
            # Stage 7: Select top M
            SelectTopPromptsStage(self.config.top_m_refine, "rigorous", **stage_kwargs),
            # Stage 8: Parallel refinement
            RefinementStage(**stage_kwargs),
        ]

    async def optimize(self) -> OptimizationResult:
        """
        Run the full optimization pipeline using the task_spec from the config.

        Returns:
            Optimization results with best prompt and detailed analysis
        """
        spec = self.config.task_spec

        start_time = time.time()
        run_id = self.storage.start_optimization_run(spec.task_description)

        # Create run-specific output directory
        run_output_dir = self.config.results_path / f"run-{run_id:04d}"
        run_output_dir.mkdir(parents=True, exist_ok=True)

        self._print_progress("=== STARTING PROMPT OPTIMIZATION ===")
        self._print_progress(f"Task: {spec.task_description}")
        self._print_progress(f"Output directory: {run_output_dir}")

        # Initialize context
        context = RunContext(task_spec=spec, output_dir=str(run_output_dir))

        # Track if we temporarily added original prompt for rigorous evaluation
        original_prompt_added_for_rigorous = False

        # Execute all stages sequentially, each updating the context
        for idx, stage in enumerate(self.stages):
            # Before Stage 6 (rigorous evaluation), add original prompt if needed
            if idx == 5:  # Stage 6 is index 5 (0-based)
                original_prompt = next(
                    (p for p in context.initial_prompts if p.is_original_system_prompt), None
                )
                if original_prompt and original_prompt not in context.top_k_prompts:
                    self._print_progress(
                        "\nAdding original prompt for rigorous evaluation (in parallel with top_k)..."
                    )
                    context.top_k_prompts.append(original_prompt)
                    original_prompt_added_for_rigorous = True

            self._print_progress(f"\n[STAGE {idx + 1}] {stage.name}")
            context = await stage.run(context)

            # After Stage 6, remove original prompt if we added it temporarily
            if idx == 5 and original_prompt_added_for_rigorous:
                self._print_progress(
                    "\nRemoving original prompt from top_k (was evaluated for comparison only)..."
                )
                context.top_k_prompts.remove(original_prompt)

        # Extract results from context
        champion = max(
            (tr.final_prompt for tr in context.refinement_tracks),
            key=lambda p: p.average_score or 0,
        )
        self._print_progress("\n=== OPTIMIZATION COMPLETE ===")
        self._print_progress(f"Champion Score: {champion.average_score:.2f}")
        self._print_progress(f"Champion Track: {champion.track_id}")

        total_time = time.time() - start_time
        total_tests = len(context.initial_prompts) * len(context.quick_tests) + len(
            context.top_k_prompts
        ) * len(context.rigorous_tests)

        self.storage.complete_optimization_run(run_id, champion.id, total_tests)

        # Find original system prompt if it was included
        original_prompt = next(
            (p for p in context.initial_prompts if p.is_original_system_prompt), None
        )

        # Get all test results for the champion
        champion_test_results = self.storage.get_prompt_evaluations(champion.id)

        # Get original prompt results (already evaluated with rigorous tests in Stage 6)
        original_rigorous_score = None
        original_test_results = []
        if original_prompt:
            # Original prompt was already evaluated with rigorous tests in Stage 6
            # (either as part of top_k or temporarily added for comparison)
            original_rigorous_score = original_prompt.average_score
            original_test_results = self.storage.get_prompt_evaluations(original_prompt.id)

        return OptimizationResult(
            run_id=run_id,
            output_dir=str(run_output_dir),
            best_prompt=champion,
            all_tracks=context.refinement_tracks,
            initial_prompts=context.initial_prompts,
            stage1_top5=context.top_k_prompts,
            stage2_top3=context.top_m_prompts,
            total_tests_run=total_tests,
            total_time_seconds=total_time,
            quick_tests=context.quick_tests,
            rigorous_tests=context.rigorous_tests,
            original_system_prompt=original_prompt,
            original_system_prompt_rigorous_score=original_rigorous_score,
            original_system_prompt_test_results=original_test_results,
            champion_test_results=champion_test_results,
        )

    def _print_progress(self, message: str, end: str = "\n") -> None:
        """Print progress if verbose mode is enabled."""
        if self.config.verbose:
            print(message, end=end, flush=True)
