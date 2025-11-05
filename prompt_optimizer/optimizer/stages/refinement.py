"""Refinement stage: Iteratively improve top candidates in parallel tracks."""

import asyncio
import uuid

from agents import Runner

from prompt_optimizer.agents.refiner_agent import create_refiner_agent
from prompt_optimizer.optimizer.base_stage import BaseStage
from prompt_optimizer.optimizer.context import RunContext
from prompt_optimizer.optimizer.utils.evaluation import evaluate_prompt
from prompt_optimizer.types import PromptCandidate, RefinementTrackResult


class RefinementStage(BaseStage):
    """Stage 8: Parallel refinement tracks with iterative improvements."""

    @property
    def name(self) -> str:
        """Return the stage name."""
        return "Parallel Refinement"

    async def run(self, context: RunContext) -> RunContext:
        """
        Refine top prompts in parallel tracks.

        Args:
            context: Run context with top_m_prompts and rigorous_tests

        Returns:
            Updated context with refinement_tracks populated
        """
        prompts = context.top_m_prompts
        self._print_progress(f"Launching {len(prompts)} parallel refinement tracks...")

        # Run tracks in parallel
        tasks = [
            self._refinement_track(prompt, context, track_id=i)
            for i, prompt in enumerate(prompts)
        ]
        track_results = await asyncio.gather(*tasks)

        context.refinement_tracks = list(track_results)
        return context

    async def _refinement_track(
        self,
        initial_prompt: PromptCandidate,
        context: RunContext,
        track_id: int,
    ) -> RefinementTrackResult:
        """Run a single refinement track with iterative improvements."""
        rigorous_tests = context.rigorous_tests
        task_spec = context.task_spec

        initial_prompt.track_id = track_id
        current_prompt = initial_prompt
        best_score = current_prompt.average_score or 0
        no_improvement_count = 0
        iterations = [current_prompt]
        score_progression = [best_score]

        self._print_progress(f"\n  Track {track_id}: Starting (score={best_score:.2f})")

        for iteration in range(1, self.config.max_iterations_per_track + 1):
            # Analyze weaknesses
            weaknesses = await self._analyze_weaknesses(current_prompt)
            failed_tests = weaknesses.get("failed_tests", [])
            weakness_description = weaknesses.get("description", "")

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
                refined_prompt, rigorous_tests, task_spec, self.config, self.model_client, self.storage
            )
            refined_prompt.average_score = new_score
            self.storage.save_prompt(refined_prompt)

            iterations.append(refined_prompt)
            score_progression.append(new_score)

            # Check for improvement
            improvement = (new_score - best_score) / best_score if best_score > 0 else 0

            if improvement >= self.config.convergence_threshold:
                current_prompt = refined_prompt
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

        return RefinementTrackResult(
            track_id=track_id,
            initial_prompt=initial_prompt,
            final_prompt=current_prompt,
            iterations=iterations,
            score_progression=score_progression,
            improvement=final_improvement,
        )

    async def _analyze_weaknesses(self, prompt: PromptCandidate) -> dict:
        """Analyze a prompt's weaknesses based on evaluation results."""
        # Get evaluations from storage
        test_results = self.storage.get_prompt_evaluations(prompt.id)

        # Find failures (overall score < 7)
        failures = [tr for tr in test_results if tr.evaluation.overall < 7.0]

        if not failures:
            return {"description": "No significant weaknesses found", "failed_tests": []}

        # Summarize failure patterns
        failed_test_descriptions = [
            f"Test {tr.test_case_id}: {tr.evaluation.reasoning}" for tr in failures[:5]
        ]

        weakness_description = (
            f"Found {len(failures)} weak test cases. "
            f"Common issues: {', '.join(set(tr.evaluation.reasoning for tr in failures[:3]))}"
        )

        return {
            "description": weakness_description,
            "failed_tests": failed_test_descriptions,
        }

    def _parse_refined_prompt(self, agent_output) -> str:
        """Parse refined prompt from agent output."""
        return agent_output.improved_prompt
