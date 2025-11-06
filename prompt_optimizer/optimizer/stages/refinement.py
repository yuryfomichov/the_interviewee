"""Refinement stage: Iteratively improve top candidates in parallel tracks."""

import asyncio
import uuid

from agents import Runner

from prompt_optimizer.agents.refiner_agent import RefinedPromptOutput, create_refiner_agent
from prompt_optimizer.optimizer.base_stage import BaseStage
from prompt_optimizer.optimizer.context import RunContext
from prompt_optimizer.optimizer.utils.evaluation import evaluate_prompt
from prompt_optimizer.storage import PromptConverter, TestCaseConverter, WeaknessAnalysisConverter
from prompt_optimizer.types import PromptCandidate, WeaknessAnalysis


class RefinementStage(BaseStage):
    """Stage 8: Parallel refinement tracks with iterative improvements."""

    @property
    def name(self) -> str:
        """Return the stage name."""
        return "Refinement"

    async def _run_async(self, context: RunContext) -> RunContext:
        """
        Refine top prompts in parallel tracks.

        Args:
            context: Run context with database access

        Returns:
            Updated context (refined prompts saved to database)
        """
        # Query top M prompts from rigorous stage
        db_prompts = context.prompt_repo.get_top_k(context.run_id, "rigorous", self.config.top_m_refine)
        prompts = [PromptConverter.from_db(p) for p in db_prompts]

        self._print_progress(f"Launching {len(prompts)} parallel refinement tracks...")

        # Create global semaphore for all refinement evaluations
        semaphore = asyncio.Semaphore(self.config.max_concurrent_evaluations)

        # Run tracks in parallel
        tasks = [
            self._refinement_track(prompt, context, track_id=i, semaphore=semaphore)
            for i, prompt in enumerate(prompts)
        ]
        await asyncio.gather(*tasks)

        return context

    async def _run_sync(self, context: RunContext) -> RunContext:
        """
        Refine top prompts sequentially.

        Args:
            context: Run context with database access

        Returns:
            Updated context (refined prompts saved to database)
        """
        # Query top M prompts from rigorous stage
        db_prompts = context.prompt_repo.get_top_k(context.run_id, "rigorous", self.config.top_m_refine)
        prompts = [PromptConverter.from_db(p) for p in db_prompts]

        self._print_progress(f"Running {len(prompts)} refinement tracks sequentially...")

        for i, prompt in enumerate(prompts):
            self._print_progress(f"\nStarting refinement track {i + 1}/{len(prompts)}...")
            await self._refinement_track(prompt, context, track_id=i, semaphore=None)

        return context

    async def _refinement_track(
        self,
        initial_prompt: PromptCandidate,
        context: RunContext,
        track_id: int,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        """Run a single refinement track with iterative improvements."""
        # Query rigorous tests from database
        db_tests = context.test_repo.get_by_stage(context.run_id, "rigorous")
        rigorous_tests = [TestCaseConverter.from_db(t) for t in db_tests]
        task_spec = context.task_spec

        # Set track ID and save initial prompt with track info
        initial_prompt.track_id = track_id
        db_initial = PromptConverter.to_db(initial_prompt, context.run_id)
        context.prompt_repo.save(db_initial)

        current_prompt = initial_prompt
        best_score = current_prompt.average_score or 0

        no_improvement_count = 0

        self._print_progress(f"\n  Track {track_id}: Starting (score={best_score:.2f})")

        parent_prompt_id = initial_prompt.id

        for iteration in range(1, self.config.max_iterations_per_track + 1):
            # Analyze weaknesses
            weaknesses = await self._analyze_weaknesses(current_prompt, context)
            failed_tests = weaknesses.get("failed_tests", [])
            failed_test_ids = weaknesses.get("failed_test_ids", [])
            weakness_description = weaknesses.get("description", "")

            # Store weakness analysis in database
            weakness_analysis = WeaknessAnalysis(
                iteration=iteration - 1,  # The iteration where this weakness was found
                description=weakness_description,
                failed_test_ids=failed_test_ids,
                failed_test_descriptions=failed_tests,
            )
            db_weakness = WeaknessAnalysisConverter.to_db(weakness_analysis, current_prompt.id)
            context._session.add(db_weakness)
            context._session.commit()

            # Generate refinement
            refiner = create_refiner_agent(
                self.config.refiner_llm,
                task_spec,
                current_prompt.prompt_text,
                weakness_description,
                failed_tests,
                iteration,
            )
            refinement_result = await Runner.run(
                refiner, f"Refine prompt (track {track_id}, iteration {iteration})"
            )
            refined_text = self._parse_refined_prompt(refinement_result.final_output)

            # Create new candidate
            refined_prompt = PromptCandidate(
                id=f"track{track_id}_iter{iteration}_{uuid.uuid4().hex[:8]}",
                prompt_text=refined_text,
                stage="refined",
                iteration=iteration,
                track_id=track_id,
            )

            # Re-evaluate
            new_score = await evaluate_prompt(
                refined_prompt,
                rigorous_tests,
                task_spec,
                self.config,
                self.model_client,
                context,
                parallel=True,
                semaphore=semaphore,
            )
            refined_prompt.average_score = new_score

            # Save to database with parent linkage
            db_refined = PromptConverter.to_db(refined_prompt, context.run_id)
            db_refined.parent_prompt_id = parent_prompt_id
            context.prompt_repo.save(db_refined)

            # Check for improvement
            improvement = (new_score - best_score) / best_score if best_score > 0 else 0

            if improvement >= self.config.convergence_threshold:
                current_prompt = refined_prompt
                parent_prompt_id = refined_prompt.id
                best_score = new_score
                no_improvement_count = 0
                self._print_progress(
                    f"  Track {track_id} iter {iteration}: "
                    f"{new_score:.2f} (+{improvement * 100:.1f}%) ✓"
                )
            else:
                no_improvement_count += 1
                self._print_progress(
                    f"  Track {track_id} iter {iteration}: {new_score:.2f} (no improvement)"
                )

            # Early stopping
            if no_improvement_count >= self.config.early_stopping_patience:
                self._print_progress(f"  Track {track_id}: Early stopping")
                break

        final_improvement = best_score - (initial_prompt.average_score or 0)
        self._print_progress(
            f"  Track {track_id}: Complete (final={best_score:.2f}, "
            f"improvement=+{final_improvement:.2f})"
        )

    async def _analyze_weaknesses(self, prompt: PromptCandidate, context: RunContext) -> dict:
        """Analyze a prompt's weaknesses based on evaluation results."""
        # Get evaluations from database
        db_evaluations = context.eval_repo.get_by_prompt(prompt.id)

        # Find failures (overall score < 7)
        failures = [ev for ev in db_evaluations if ev.overall_score < 7.0]

        if not failures:
            return {
                "description": "No significant weaknesses found",
                "failed_tests": [],
                "failed_test_ids": [],
            }

        # Summarize failure patterns
        failed_test_ids = [ev.test_case_id for ev in failures]
        failed_test_descriptions = [
            f"Test {ev.test_case_id}: {ev.reasoning}" for ev in failures[:5]
        ]

        weakness_description = (
            f"Found {len(failures)} weak test cases. "
            f"Common issues: {', '.join({ev.reasoning for ev in failures[:3]})}"
        )

        return {
            "description": weakness_description,
            "failed_tests": failed_test_descriptions,
            "failed_test_ids": failed_test_ids,
        }

    def _parse_refined_prompt(self, agent_output: RefinedPromptOutput) -> str:
        """Parse refined prompt from agent output."""
        return agent_output.improved_prompt
