"""Reporting stage: Collect and format final optimization results."""

import time

from prompt_optimizer.optimizer.base_stage import BaseStage
from prompt_optimizer.optimizer.context import RunContext
from prompt_optimizer.types import OptimizationResult


class ReportingStage(BaseStage):
    """Final stage to collect and format optimization results."""

    @property
    def name(self) -> str:
        """Return the stage name."""
        return "Prepare Report"

    async def _run_async(self, context: RunContext) -> RunContext:
        """
        Collect final results and create optimization report (async mode).

        Args:
            context: Run context with all pipeline results

        Returns:
            Updated context with optimization_result populated
        """
        return await self._generate_report(context)

    async def _run_sync(self, context: RunContext) -> RunContext:
        """
        Collect final results and create optimization report (sync mode).

        Args:
            context: Run context with all pipeline results

        Returns:
            Updated context with optimization_result populated
        """
        return await self._generate_report(context)

    async def _generate_report(self, context: RunContext) -> RunContext:
        """
        Generate the optimization report.

        Args:
            context: Run context with all pipeline results

        Returns:
            Updated context with optimization_result populated
        """
        # Get run metadata from context
        if context.run_id is None or context.start_time is None:
            raise ValueError("run_id and start_time must be set in context")

        # Find the champion prompt (best from refinement tracks)
        champion = max(
            (tr.final_prompt for tr in context.refinement_tracks),
            key=lambda p: p.average_score or 0,
        )

        self._print_progress("\n=== OPTIMIZATION COMPLETE ===")
        self._print_progress(f"Champion Score: {champion.average_score:.2f}")
        self._print_progress(f"Champion Track: {champion.track_id}")

        # Calculate total time and tests
        total_time = time.time() - context.start_time
        total_tests = len(context.initial_prompts) * len(context.quick_tests) + len(
            context.top_k_prompts
        ) * len(context.rigorous_tests)

        # Complete the run in storage
        self.storage.complete_optimization_run(context.run_id, champion.id, total_tests)

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

        # Create the optimization result
        result = OptimizationResult(
            run_id=context.run_id,
            output_dir=context.output_dir or "",
            best_prompt=champion,
            all_tracks=context.refinement_tracks,
            initial_prompts=context.initial_prompts,
            top_k_prompts=context.top_k_prompts,
            top_m_prompts=context.top_m_prompts,
            total_tests_run=total_tests,
            total_time_seconds=total_time,
            quick_tests=context.quick_tests,
            rigorous_tests=context.rigorous_tests,
            original_system_prompt=original_prompt,
            original_system_prompt_rigorous_score=original_rigorous_score,
            original_system_prompt_test_results=original_test_results,
            champion_test_results=champion_test_results,
        )

        # Store result in context
        context.optimization_result = result

        return context
