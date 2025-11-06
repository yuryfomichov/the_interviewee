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
    ReportingStage,
    SaveReportsStage,
    SelectTopPromptsStage,
)
from prompt_optimizer.storage import Database
from prompt_optimizer.types import OptimizationResult

logger = logging.getLogger(__name__)


class PromptOptimizer:
    """Main orchestrator for the prompt optimization pipeline."""

    def __init__(
        self,
        model_client,
        config: OptimizerConfig | None = None,
        database: Database | None = None,
    ):
        """
        Initialize the optimizer.

        Args:
            model_client: Client for testing the target model.
            config: Optimizer configuration with a populated task_spec.
            database: Database instance (creates new if None).
        """
        self.model_client = model_client
        if config is None:
            raise ValueError("OptimizerConfig with task_spec must be provided.")
        self.config = config
        self.database = database or Database(self.config.database_path)

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
            "database": self.database,
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
            # Stage 9: Prepare report
            ReportingStage(**stage_kwargs),
            # Stage 10: Save reports to disk
            SaveReportsStage(**stage_kwargs),
        ]

    async def optimize(self) -> OptimizationResult:
        """
        Run the full optimization pipeline using the task_spec from the config.

        Returns:
            Optimization results with best prompt and detailed analysis
        """
        spec = self.config.task_spec
        start_time = time.time()

        # Create a database session for this optimization run
        session = self.database.get_session()

        try:
            # Create optimization run in database
            from prompt_optimizer.storage.repositories import RunRepository

            run_repo = RunRepository(session)
            run = run_repo.create(spec.task_description)
            run_id = run.id

            # Create run-specific output directory
            run_output_dir = self.config.results_path / f"run-{run_id:04d}"
            run_output_dir.mkdir(parents=True, exist_ok=True)

            self._print_progress("=== STARTING PROMPT OPTIMIZATION ===")
            self._print_progress(f"Task: {spec.task_description}")
            self._print_progress(f"Output directory: {run_output_dir}")

            # Initialize minimal context with run metadata
            context = RunContext(
                run_id=run_id,
                task_spec=spec,
                start_time=start_time,
                output_dir=str(run_output_dir),
            )
            # Attach database session to context
            context.set_session(session)

            # Execute all stages sequentially, each updating the database
            for idx, stage in enumerate(self.stages):
                self._print_progress(f"\n[STAGE {idx + 1}] {stage.name}")
                context = await stage.run(context)
                run_repo.update_stage(run_id, stage.name)

            # Get the optimization result created by ReportingStage
            if not hasattr(context, '_optimization_result'):
                raise RuntimeError("ReportingStage did not create optimization result")

            return context._optimization_result  # type: ignore

        finally:
            # Always close the session
            session.close()

    def _print_progress(self, message: str, end: str = "\n") -> None:
        """Print progress if verbose mode is enabled."""
        if self.config.verbose:
            print(message, end=end, flush=True)
