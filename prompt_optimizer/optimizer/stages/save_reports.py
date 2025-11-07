"""Save reports stage: Save all optimization reports to disk."""

import asyncio

from prompt_optimizer.optimizer.base_stage import BaseStage
from prompt_optimizer.optimizer.context import RunContext
from prompt_optimizer.reports import (
    save_champion_prompt,
    save_champion_qa_results,
    save_champion_questions,
    save_optimization_report,
    save_original_prompt_quick_report,
    save_original_prompt_rigorous_results,
    save_prompts_json,
    save_testcases_json,
)


class SaveReportsStage(BaseStage):
    """Final stage to save all optimization reports to disk."""

    @property
    def name(self) -> str:
        """Return the stage name."""
        return "Save Reports"

    async def _run_async(self, context: RunContext) -> RunContext:
        """
        Save all reports to disk concurrently (async mode).

        Args:
            context: Run context with _optimization_result

        Returns:
            Unchanged context (reports saved to disk)
        """
        if not hasattr(context, "_optimization_result"):
            raise ValueError("_optimization_result must be set in context by ReportingStage")
        if context.output_dir is None:
            raise ValueError("output_dir must be set in context")

        result = context._optimization_result  # type: ignore
        output_dir = context.output_dir

        self._print_progress("\nSaving final reports...")

        # Save all reports concurrently
        tasks = [
            save_champion_prompt(result, output_dir),
            save_optimization_report(result, context.task_spec, output_dir),
            save_champion_questions(result, output_dir),
            save_champion_qa_results(result, output_dir),
            save_original_prompt_rigorous_results(result, output_dir),
            save_testcases_json(result, output_dir),
            save_prompts_json(result, output_dir),
        ]

        # Add original prompt quick report if original prompt exists
        if result.original_system_prompt:
            tasks.append(
                save_original_prompt_quick_report(
                    original_prompt=result.original_system_prompt,
                    quick_tests=result.quick_tests,
                    initial_prompts=result.initial_prompts,
                    top_k_prompts=result.top_k_prompts,
                    context=context,
                    output_dir=output_dir,
                )
            )

        await asyncio.gather(*tasks)

        self._print_progress("All reports saved successfully.")
        return context

    async def _run_sync(self, context: RunContext) -> RunContext:
        """
        Save all reports to disk sequentially (sync mode).

        Args:
            context: Run context with _optimization_result

        Returns:
            Unchanged context (reports saved to disk)
        """
        if not hasattr(context, "_optimization_result"):
            raise ValueError("_optimization_result must be set in context by ReportingStage")
        if context.output_dir is None:
            raise ValueError("output_dir must be set in context")

        result = context._optimization_result  # type: ignore
        output_dir = context.output_dir

        self._print_progress("\nSaving final reports...")

        # Save all reports sequentially
        await save_champion_prompt(result, output_dir)
        await save_optimization_report(result, context.task_spec, output_dir)
        await save_champion_questions(result, output_dir)
        await save_champion_qa_results(result, output_dir)
        await save_original_prompt_rigorous_results(result, output_dir)
        await save_testcases_json(result, output_dir)
        await save_prompts_json(result, output_dir)

        # Save original prompt quick report if original prompt exists
        if result.original_system_prompt:
            await save_original_prompt_quick_report(
                original_prompt=result.original_system_prompt,
                quick_tests=result.quick_tests,
                initial_prompts=result.initial_prompts,
                top_k_prompts=result.top_k_prompts,
                context=context,
                output_dir=output_dir,
            )

        self._print_progress("All reports saved successfully.")
        return context
