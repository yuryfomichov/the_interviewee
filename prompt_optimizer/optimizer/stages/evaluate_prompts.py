"""Evaluate prompts stage: Run prompts against test cases."""

import asyncio
from typing import Literal

from prompt_optimizer.optimizer.base_stage import BaseStage
from prompt_optimizer.optimizer.context import RunContext
from prompt_optimizer.optimizer.utils.evaluation import evaluate_prompt
from prompt_optimizer.storage import PromptConverter, TestCaseConverter


class EvaluatePromptsStage(BaseStage):
    """Evaluate prompts against test cases."""

    stage_name: Literal["quick_filter", "rigorous"]

    def __init__(self, stage_name: Literal["quick_filter", "rigorous"], top_k: int | None = None, *args, **kwargs):
        """
        Initialize evaluation stage.

        Args:
            stage_name: Name for the evaluation stage (e.g., "quick_filter", "rigorous")
            top_k: Optional limit on number of prompts to evaluate (for rigorous stage)
            *args, **kwargs: Passed to BaseStage
        """
        super().__init__(*args, **kwargs)
        self.stage_name = stage_name
        self.top_k = top_k

    @property
    def name(self) -> str:
        """Return the stage name."""
        return f"Evaluate ({self.stage_name.replace('_', ' ').title()})"

    async def _run_async(self, context: RunContext) -> RunContext:
        """
        Evaluate prompts against test cases in parallel with global concurrency control.

        Args:
            context: Run context with database access

        Returns:
            Updated context (prompts updated with scores in database)
        """
        # Query prompts and tests from database
        prompts, tests, original_prompt_for_comparison = self._get_prompts_and_tests(
            context, execution_mode="parallel"
        )

        total_evaluations = len(prompts) * len(tests)
        self._print_progress(
            f"Evaluating {len(prompts)} prompts × {len(tests)} tests "
            f"({total_evaluations} total evaluations, max {self.config.max_concurrent_evaluations} concurrent)..."
        )

        # Create global semaphore shared across ALL evaluations (prompts × tests)
        semaphore = asyncio.Semaphore(self.config.max_concurrent_evaluations)

        # Evaluate all prompts in parallel (semaphore controls test-level concurrency)
        eval_tasks = [
            evaluate_prompt(
                prompt,
                tests,
                context.task_spec,
                self.config,
                self.model_client,
                context,
                parallel=True,
                semaphore=semaphore,
            )
            for prompt in prompts
        ]

        scores = await asyncio.gather(*eval_tasks)

        # Update prompts with scores and save to database
        self._update_and_save_prompt_scores(context, prompts, scores, original_prompt_for_comparison)

        # Report original prompt comparison if applicable
        self._report_original_prompt_comparison(original_prompt_for_comparison)

        self._print_progress(
            f"Evaluation complete (scores: {[f'{p.average_score:.2f}' for p in prompts[:5]]}...)"
        )

        return context

    async def _run_sync(self, context: RunContext) -> RunContext:
        """
        Evaluate prompts against test cases sequentially (no concurrency).

        Args:
            context: Run context with database access

        Returns:
            Updated context (prompts updated with scores in database)
        """
        # Query prompts and tests from database
        prompts, tests, original_prompt_for_comparison = self._get_prompts_and_tests(
            context, execution_mode="sequential"
        )

        total_evaluations = len(prompts) * len(tests)
        self._print_progress(
            f"Evaluating {len(prompts)} prompts × {len(tests)} tests "
            f"({total_evaluations} total evaluations, sequential mode)..."
        )

        # No semaphore in sequential mode - everything runs one at a time
        scores = []
        for i, prompt in enumerate(prompts, 1):
            self._print_progress(f"  Evaluating prompt {i}/{len(prompts)}...")
            avg_score = await evaluate_prompt(
                prompt,
                tests,
                context.task_spec,
                self.config,
                self.model_client,
                context,
                parallel=False,
                semaphore=None,
            )
            scores.append(avg_score)

        # Update prompts with scores and save to database
        self._update_and_save_prompt_scores(context, prompts, scores, original_prompt_for_comparison)

        # Report original prompt comparison if applicable
        self._report_original_prompt_comparison(original_prompt_for_comparison)

        self._print_progress(
            f"Evaluation complete (scores: {[f'{p.average_score:.2f}' for p in prompts[:5]]}...)"
        )

        return context

    def _get_prompts_and_tests(
        self, context: RunContext, execution_mode: Literal["parallel", "sequential"]
    ) -> tuple[list, list, object | None]:
        """
        Query and convert prompts and tests from database.

        Args:
            context: Run context with database access
            execution_mode: "parallel" or "sequential" for logging purposes

        Returns:
            Tuple of (prompts, tests, original_prompt_for_comparison)
        """
        original_prompt_for_comparison = None

        if self.stage_name == "quick_filter":
            # Get all prompts from initial stage
            db_prompts = context.prompt_repo.get_by_stage(context.run_id, "initial")
            db_tests = context.test_repo.get_by_stage(context.run_id, "quick")
        else:  # rigorous
            # Get top K prompts from quick_filter stage (limit if top_k specified)
            if self.top_k is not None:
                db_prompts = context.prompt_repo.get_top_k(context.run_id, "quick_filter", self.top_k)
            else:
                db_prompts = context.prompt_repo.get_by_stage(context.run_id, "quick_filter")
            db_tests = context.test_repo.get_by_stage(context.run_id, "rigorous")

            # For rigorous evaluation, check if we need to evaluate original prompt for comparison
            original_db_prompt = context.prompt_repo.get_original_prompt(context.run_id)
            if original_db_prompt and original_db_prompt not in db_prompts:
                mode_text = execution_mode + " with top_k"
                self._print_progress(
                    f"Adding original prompt for rigorous evaluation ({mode_text})..."
                )
                db_prompts.append(original_db_prompt)
                original_prompt_for_comparison = original_db_prompt

        # Convert to Pydantic models for evaluation
        prompts = [PromptConverter.from_db(p) for p in db_prompts]
        tests = [TestCaseConverter.from_db(t) for t in db_tests]

        return prompts, tests, original_prompt_for_comparison

    def _update_and_save_prompt_scores(
        self, context: RunContext, prompts: list, scores: list[float], comparison_prompt = None
    ) -> None:
        """
        Update prompts with scores and save to database.

        Args:
            context: Run context with database access
            prompts: List of prompt candidates
            scores: List of average scores corresponding to prompts
            comparison_prompt: Optional DB prompt that was added for comparison only (should not advance)
        """
        for prompt, avg_score in zip(prompts, scores, strict=True):
            prompt.average_score = avg_score

            # Check if this is a comparison-only prompt (should not advance)
            is_comparison_only = (
                comparison_prompt is not None
                and prompt.id == comparison_prompt.id
            )

            if is_comparison_only:
                # This prompt was added for comparison only - don't update its stage
                # Get existing stage from DB
                existing_db_prompt = context.prompt_repo.get_by_id(prompt.id)
                if existing_db_prompt:
                    prompt.stage = existing_db_prompt.stage  # Keep at current stage
                # Store the rigorous score for reporting
                prompt.rigorous_score = avg_score
            else:
                # Normal flow: update stage for all prompts (including original if it was selected)
                prompt.stage = self.stage_name
                # Store scores in dedicated fields for original prompt
                if prompt.is_original_system_prompt:
                    if self.stage_name == "quick_filter":
                        prompt.quick_score = avg_score
                    else:  # rigorous
                        prompt.rigorous_score = avg_score

            # Save to database
            db_prompt = PromptConverter.to_db(prompt, context.run_id)
            context.prompt_repo.save(db_prompt)

    def _report_original_prompt_comparison(self, original_prompt_for_comparison: object | None) -> None:
        """
        Report original prompt comparison if applicable.

        Args:
            original_prompt_for_comparison: Database prompt object or None
        """
        if original_prompt_for_comparison:
            original_pydantic = PromptConverter.from_db(original_prompt_for_comparison)
            self._print_progress(
                f"Original prompt rigorous score: {original_pydantic.rigorous_score:.2f} "
                "(evaluated for comparison only, not advancing to next stage)"
            )
