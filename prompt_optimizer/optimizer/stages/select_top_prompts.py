"""Select top prompts stage: Filter best performing candidates."""

from prompt_optimizer.optimizer.base_stage import BaseStage
from prompt_optimizer.optimizer.context import RunContext
from prompt_optimizer.reports import save_original_prompt_quick_report
from prompt_optimizer.storage import PromptConverter


class SelectTopPromptsStage(BaseStage):
    """Select top N performing prompts."""

    def __init__(self, top_n: int, selection_type: str, *args, **kwargs):
        """
        Initialize selection stage.

        Args:
            top_n: Number of top prompts to select
            selection_type: "quick" or "rigorous" to determine which stage to query from
            *args, **kwargs: Passed to BaseStage
        """
        super().__init__(*args, **kwargs)
        self.top_n = top_n
        self.selection_type = selection_type

    @property
    def name(self) -> str:
        """Return the stage name."""
        return f"Select Top {self.top_n}"

    async def _run_async(self, context: RunContext) -> RunContext:
        """
        Select top N prompts by score (async mode).

        Args:
            context: Run context with database access

        Returns:
            Updated context (top prompts already in database with scores)
        """
        # Query top N prompts from database based on selection type
        if self.selection_type == "quick":
            # Get top K from quick_filter stage
            db_top_prompts = context.prompt_repo.get_top_k(
                context.run_id, "quick_filter", self.top_n
            )
        else:  # rigorous
            # Get top M from rigorous stage
            db_top_prompts = context.prompt_repo.get_top_k(context.run_id, "rigorous", self.top_n)

        # Convert to Pydantic for display
        top_prompts = [PromptConverter.from_db(p) for p in db_top_prompts]

        self._print_progress(
            f"\nTop {self.top_n} prompts selected "
            f"(scores: {[f'{p.average_score:.2f}' for p in top_prompts]})"
        )

        # For quick selection, save original prompt quick test report if available
        if self.selection_type == "quick" and context.output_dir:
            original_db_prompt = context.prompt_repo.get_original_prompt(context.run_id)
            if original_db_prompt:
                self._print_progress("\nSaving original prompt quick test report...")

                # Query data needed for report
                # Note: After quick_filter evaluation, prompts have stage="quick_filter", not "initial"
                initial_db_prompts = context.prompt_repo.get_by_stage(
                    context.run_id, "quick_filter"
                )
                quick_db_tests = context.test_repo.get_by_stage(context.run_id, "quick")

                # Convert to Pydantic
                from prompt_optimizer.storage import TestCaseConverter

                original_prompt = PromptConverter.from_db(original_db_prompt)
                initial_prompts = [PromptConverter.from_db(p) for p in initial_db_prompts]
                quick_tests = [TestCaseConverter.from_db(t) for t in quick_db_tests]

                save_original_prompt_quick_report(
                    original_prompt=original_prompt,
                    quick_tests=quick_tests,
                    initial_prompts=initial_prompts,
                    top_k_prompts=top_prompts,
                    context=context,
                    output_dir=context.output_dir,
                )

        return context

    async def _run_sync(self, context: RunContext) -> RunContext:
        """
        Select top N prompts by score (sync mode - same as async for this stage).

        Args:
            context: Run context with database access

        Returns:
            Updated context (top prompts already in database with scores)
        """
        # This stage is purely computational, no difference between sync and async
        return await self._run_async(context)
