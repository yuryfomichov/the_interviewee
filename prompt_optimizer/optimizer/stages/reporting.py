"""Reporting stage: Collect and format final optimization results."""

import time

from prompt_optimizer.optimizer.base_stage import BaseStage
from prompt_optimizer.optimizer.context import RunContext
from prompt_optimizer.storage import EvaluationConverter, PromptConverter, TestCaseConverter
from prompt_optimizer.types import OptimizationResult, RefinementTrackResult


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
            context: Run context with database access

        Returns:
            Updated context
        """
        return await self._generate_report(context)

    async def _run_sync(self, context: RunContext) -> RunContext:
        """
        Collect final results and create optimization report (sync mode).

        Args:
            context: Run context with database access

        Returns:
            Updated context
        """
        return await self._generate_report(context)

    async def _generate_report(self, context: RunContext) -> RunContext:
        """
        Generate the optimization report by querying all data from database.

        Args:
            context: Run context with database access

        Returns:
            Updated context
        """
        # Query all data from database
        self._print_progress("\nCollecting results from database...")

        # Get all refined prompts (best from each track)
        db_refined_prompts = context.prompt_repo.get_by_stage(context.run_id, "refined")

        # Find champion (best overall score from all refined prompts)
        champion_db = max(db_refined_prompts, key=lambda p: p.average_score or 0)
        champion = PromptConverter.from_db(champion_db)

        self._print_progress("\n=== OPTIMIZATION COMPLETE ===")
        self._print_progress(f"Champion Score: {champion.average_score:.2f}")
        self._print_progress(f"Champion Track: {champion.track_id}")

        # Calculate total time
        total_time = time.time() - context.start_time

        # Count total evaluations
        total_tests = context.eval_repo.count_for_run(context.run_id)

        # Complete the run in database
        context.run_repo.complete(context.run_id, champion.id, total_tests, total_time)

        # Query all data needed for OptimizationResult
        db_initial_prompts = context.prompt_repo.get_by_stage(context.run_id, "initial")
        db_top_k_prompts = context.prompt_repo.get_by_stage(context.run_id, "quick_filter")
        db_top_m_prompts = context.prompt_repo.get_by_stage(context.run_id, "rigorous")
        db_quick_tests = context.test_repo.get_by_stage(context.run_id, "quick")
        db_rigorous_tests = context.test_repo.get_by_stage(context.run_id, "rigorous")

        # Convert to Pydantic models
        initial_prompts = [PromptConverter.from_db(p) for p in db_initial_prompts]
        top_k_prompts = [PromptConverter.from_db(p) for p in db_top_k_prompts]
        top_m_prompts = [PromptConverter.from_db(p) for p in db_top_m_prompts]
        quick_tests = [TestCaseConverter.from_db(t) for t in db_quick_tests]
        rigorous_tests = [TestCaseConverter.from_db(t) for t in db_rigorous_tests]

        # Build refinement track results
        refinement_tracks = []
        for track_id in range(self.config.top_m_refine):
            track_prompts = context.prompt_repo.get_by_track(context.run_id, track_id)
            if not track_prompts:
                continue

            pydantic_prompts = [PromptConverter.from_db(p) for p in track_prompts]
            initial_prompt = pydantic_prompts[0]
            final_prompt = max(pydantic_prompts, key=lambda p: p.average_score or 0)
            score_progression = [p.average_score or 0 for p in pydantic_prompts]
            improvement = (final_prompt.average_score or 0) - (initial_prompt.average_score or 0)

            # Get weaknesses for this track (from initial prompt of track)
            from prompt_optimizer.storage import WeaknessAnalysisConverter
            db_weaknesses = []
            for p in track_prompts:
                if hasattr(p, 'weaknesses'):
                    db_weaknesses.extend(p.weaknesses)
            weaknesses_history = [WeaknessAnalysisConverter.from_db(w) for w in db_weaknesses]

            track_result = RefinementTrackResult(
                track_id=track_id,
                initial_prompt=initial_prompt,
                final_prompt=final_prompt,
                iterations=pydantic_prompts,
                score_progression=score_progression,
                improvement=improvement,
                weaknesses_history=weaknesses_history,
            )
            refinement_tracks.append(track_result)

        # Get original prompt if it exists
        original_db_prompt = context.prompt_repo.get_original_prompt(context.run_id)
        original_prompt = PromptConverter.from_db(original_db_prompt) if original_db_prompt else None
        original_rigorous_score = original_prompt.rigorous_score if original_prompt else None

        # Get champion evaluations
        db_champion_evals = context.eval_repo.get_by_prompt(champion.id)
        champion_test_results = [EvaluationConverter.from_db(ev) for ev in db_champion_evals]

        # Get original prompt evaluations
        original_test_results = []
        if original_db_prompt:
            db_original_evals = context.eval_repo.get_by_prompt(original_db_prompt.id)
            original_test_results = [EvaluationConverter.from_db(ev) for ev in db_original_evals]

        # Create the optimization result
        result = OptimizationResult(
            run_id=context.run_id,
            output_dir=context.output_dir or "",
            best_prompt=champion,
            all_tracks=refinement_tracks,
            initial_prompts=initial_prompts,
            top_k_prompts=top_k_prompts,
            top_m_prompts=top_m_prompts,
            total_tests_run=total_tests,
            total_time_seconds=total_time,
            quick_tests=quick_tests,
            rigorous_tests=rigorous_tests,
            original_system_prompt=original_prompt,
            original_system_prompt_rigorous_score=original_rigorous_score,
            original_system_prompt_test_results=original_test_results,
            champion_test_results=champion_test_results,
        )

        # Store result temporarily so orchestrator can access it
        # We'll use a private attribute since context is minimal
        context._optimization_result = result  # type: ignore

        return context
