"""Select top prompts stage: Filter best performing candidates."""

from prompt_optimizer.optimizer.base_stage import BaseStage
from prompt_optimizer.optimizer.context import RunContext
from prompt_optimizer.reporter import save_original_prompt_quick_report


class SelectTopPromptsStage(BaseStage):
    """Select top N performing prompts."""

    def __init__(self, top_n: int, selection_type: str, *args, **kwargs):
        """
        Initialize selection stage.

        Args:
            top_n: Number of top prompts to select
            selection_type: "quick" or "rigorous" to determine which field to update
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
            context: Run context with prompts

        Returns:
            Updated context with top_k_prompts or top_m_prompts populated
        """
        # Determine which prompts to select from
        if self.selection_type == "quick":
            prompts = context.initial_prompts.copy()
        else:  # rigorous
            prompts = context.top_k_prompts.copy()

        prompts.sort(key=lambda p: p.average_score or 0, reverse=True)
        top_prompts = prompts[: self.top_n]

        self._print_progress(
            f"\nTop {self.top_n} prompts selected "
            f"(scores: {[f'{p.average_score:.2f}' for p in top_prompts]})"
        )

        # Update context
        if self.selection_type == "quick":
            context.top_k_prompts = top_prompts

            # Save original prompt quick test report if available
            if self.output_dir:
                original_prompt = next(
                    (p for p in context.initial_prompts if p.is_original_system_prompt), None
                )
                if original_prompt:
                    self._print_progress("\nSaving original prompt quick test report...")
                    save_original_prompt_quick_report(
                        original_prompt=original_prompt,
                        quick_tests=context.quick_tests,
                        initial_prompts=context.initial_prompts,
                        top_k_prompts=context.top_k_prompts,
                        storage=self.storage,
                        output_dir=self.output_dir,
                    )
        else:  # rigorous
            context.top_m_prompts = top_prompts

        return context

    async def _run_sync(self, context: RunContext) -> RunContext:
        """
        Select top N prompts by score (sync mode - same as async for this stage).

        Args:
            context: Run context with prompts

        Returns:
            Updated context with top_k_prompts or top_m_prompts populated
        """
        # This stage is purely computational, no difference between sync and async
        return await self._run_async(context)
