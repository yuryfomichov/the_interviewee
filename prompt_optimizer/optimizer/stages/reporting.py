"""Reporting stage: Collect and format final optimization results."""

import time

from prompt_optimizer.optimizer.base_stage import BaseStage
from prompt_optimizer.optimizer.context import RunContext
from prompt_optimizer.schemas import (
    OptimizationResult,
    PromptCandidate,
    RefinementTrackResult,
    TestResult,
)
from prompt_optimizer.storage import (
    EvaluationConverter,
    PromptConverter,
    TestCaseConverter,
    WeaknessAnalysisConverter,
)


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
            Updated context with _optimization_result set
        """
        self._print_progress("\nCollecting results from database...")

        # Find champion prompt
        champion = self._find_champion(context)

        self._print_progress("\n=== OPTIMIZATION COMPLETE ===")
        self._print_progress(f"Champion Score: {champion.average_score:.2f}")
        self._print_progress(f"Champion Track: {champion.track_id}")

        # Calculate metrics and complete run
        total_time = time.time() - context.start_time
        total_tests = context.eval_repo.count_for_run(context.run_id)
        context.run_repo.complete(context.run_id, champion.id, total_tests, total_time)

        # Query all prompts and tests
        prompts_data = self._query_all_prompts_and_tests(context)

        # Build refinement tracks
        refinement_tracks = self._build_refinement_tracks(context)

        # Get original prompt data
        original_prompt, original_rigorous_score, original_test_results = (
            self._get_original_prompt_data(context)
        )

        # Get champion evaluations
        champion_test_results = self._get_evaluation_results(context, champion.id)

        # Create the optimization result
        result = OptimizationResult(
            run_id=context.run_id,
            output_dir=context.output_dir or "",
            best_prompt=champion,
            all_tracks=refinement_tracks,
            initial_prompts=prompts_data["initial_prompts"],
            top_k_prompts=prompts_data["top_k_prompts"],
            top_m_prompts=prompts_data["top_m_prompts"],
            total_tests_run=total_tests,
            total_time_seconds=total_time,
            quick_tests=prompts_data["quick_tests"],
            rigorous_tests=prompts_data["rigorous_tests"],
            original_system_prompt=original_prompt,
            original_system_prompt_rigorous_score=original_rigorous_score,
            original_system_prompt_test_results=original_test_results,
            champion_test_results=champion_test_results,
        )

        # Store result temporarily so orchestrator can access it
        context._optimization_result = result  # type: ignore

        return context

    def _find_champion(self, context: RunContext) -> PromptCandidate:
        """
        Find the champion prompt from all refined prompts.

        Args:
            context: Run context with database access

        Returns:
            Champion prompt with highest score
        """
        # Get ALL prompts from refinement tracks (stages: rigorous or refined)
        # - track_id is not None (was selected for a refinement track)
        # - stage in ["rigorous", "refined"] (iteration 0 is "rigorous", iter 1+ is "refined")
        # This excludes prompts that only went through quick filter
        all_prompts = context.prompt_repo.get_all_for_run(context.run_id)
        track_prompts = [
            p for p in all_prompts
            if p.track_id is not None and p.stage in ["rigorous", "refined"]
        ]

        if not track_prompts:
            raise ValueError("No refinement track prompts found!")

        champion_db = max(track_prompts, key=lambda p: p.average_score or 0)
        return PromptConverter.from_db(champion_db)

    def _query_all_prompts_and_tests(self, context: RunContext) -> dict:
        """
        Query all prompts and tests from database and convert to Pydantic models.

        Args:
            context: Run context with database access

        Returns:
            Dictionary with converted prompts and tests
        """
        # Query from database
        # Get all initially generated prompts (iteration=0, went through quick evaluation)
        # Note: Can't use stage="initial" because prompts advance to "quick_filter" after evaluation
        all_prompts = context.prompt_repo.get_all_for_run(context.run_id)
        db_initial_prompts = [
            p for p in all_prompts if p.iteration == 0 and p.quick_score is not None
        ]
        db_top_k_prompts = context.prompt_repo.get_by_stage(context.run_id, "quick_filter")
        db_top_m_prompts = context.prompt_repo.get_by_stage(context.run_id, "rigorous")
        db_quick_tests = context.test_repo.get_by_stage(context.run_id, "quick")
        db_rigorous_tests = context.test_repo.get_by_stage(context.run_id, "rigorous")

        # Convert to Pydantic models
        return {
            "initial_prompts": [PromptConverter.from_db(p) for p in db_initial_prompts],
            "top_k_prompts": [PromptConverter.from_db(p) for p in db_top_k_prompts],
            "top_m_prompts": [PromptConverter.from_db(p) for p in db_top_m_prompts],
            "quick_tests": [TestCaseConverter.from_db(t) for t in db_quick_tests],
            "rigorous_tests": [TestCaseConverter.from_db(t) for t in db_rigorous_tests],
        }

    def _build_refinement_tracks(self, context: RunContext) -> list[RefinementTrackResult]:
        """
        Build refinement track results from database prompts.

        Args:
            context: Run context with database access

        Returns:
            List of refinement track results
        """
        refinement_tracks = []

        for track_id in range(self.config.top_m_refine):
            track_prompts = context.prompt_repo.get_by_track(context.run_id, track_id)
            if not track_prompts:
                continue

            # Convert to Pydantic and extract metrics
            pydantic_prompts = [PromptConverter.from_db(p) for p in track_prompts]
            initial_prompt = pydantic_prompts[0]
            # final_prompt should be the LAST iteration, not the best scoring one
            # (track comparison shows progression: initial → final)
            final_prompt = pydantic_prompts[-1]
            score_progression = [p.average_score or 0 for p in pydantic_prompts]
            improvement = (final_prompt.average_score or 0) - (initial_prompt.average_score or 0)

            # Get weaknesses for this track
            db_weaknesses = []
            for p in track_prompts:
                if hasattr(p, "weaknesses"):
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

        return refinement_tracks

    def _get_original_prompt_data(
        self, context: RunContext
    ) -> tuple[PromptCandidate | None, float | None, list[TestResult]]:
        """
        Get original prompt and its evaluation data.

        Args:
            context: Run context with database access

        Returns:
            Tuple of (original_prompt, rigorous_score, test_results)
        """
        original_db_prompt = context.prompt_repo.get_original_prompt(context.run_id)

        if not original_db_prompt:
            return None, None, []

        original_prompt = PromptConverter.from_db(original_db_prompt)
        original_rigorous_score = original_prompt.rigorous_score
        original_test_results = self._get_evaluation_results(context, original_db_prompt.id)

        return original_prompt, original_rigorous_score, original_test_results

    def _get_evaluation_results(self, context: RunContext, prompt_id: str) -> list[TestResult]:
        """
        Get all evaluation results for a specific prompt.

        Args:
            context: Run context with database access
            prompt_id: Prompt ID to get evaluations for

        Returns:
            List of test results
        """
        db_evaluations = context.eval_repo.get_by_prompt(prompt_id)
        return [EvaluationConverter.from_db(ev) for ev in db_evaluations]
