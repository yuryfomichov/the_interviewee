"""Save reports stage: Save all optimization reports to disk."""

from prompt_optimizer.optimizer.base_stage import BaseStage
from prompt_optimizer.optimizer.context import RunContext
from prompt_optimizer.reports import (
    save_champion_prompt,
    save_champion_qa_results,
    save_champion_questions,
    save_optimization_report,
    save_original_prompt_rigorous_results,
)


class SaveReportsStage(BaseStage):
    """Final stage to save all optimization reports to disk."""

    @property
    def name(self) -> str:
        """Return the stage name."""
        return "Save Reports"

    async def _run_async(self, context: RunContext) -> RunContext:
        """
        Save all reports to disk (async mode).

        Args:
            context: Run context with optimization_result

        Returns:
            Unchanged context (reports saved to disk)
        """
        return await self._save_reports(context)

    async def _run_sync(self, context: RunContext) -> RunContext:
        """
        Save all reports to disk (sync mode).

        Args:
            context: Run context with optimization_result

        Returns:
            Unchanged context (reports saved to disk)
        """
        return await self._save_reports(context)

    async def _save_reports(self, context: RunContext) -> RunContext:
        """
        Save all reports to disk.

        Args:
            context: Run context with optimization_result

        Returns:
            Unchanged context (reports saved to disk)
        """
        # Validate we have everything we need
        if context.optimization_result is None:
            raise ValueError("optimization_result must be set in context")

        if context.output_dir is None:
            raise ValueError("output_dir must be set in context")

        result = context.optimization_result
        output_dir = context.output_dir

        self._print_progress("\nSaving final reports...")

        # Save all reports
        save_champion_prompt(result, output_dir=output_dir)
        save_optimization_report(result, context.task_spec, output_dir=output_dir)
        save_champion_questions(result, output_dir=output_dir)
        save_champion_qa_results(result, output_dir=output_dir)
        save_original_prompt_rigorous_results(result, output_dir=output_dir)

        self._print_progress("All reports saved successfully.")

        return context
