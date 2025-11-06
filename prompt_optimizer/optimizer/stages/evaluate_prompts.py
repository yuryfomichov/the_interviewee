"""Evaluate prompts stage: Run prompts against test cases."""

import asyncio
from typing import Literal

from prompt_optimizer.optimizer.base_stage import BaseStage
from prompt_optimizer.optimizer.context import RunContext
from prompt_optimizer.optimizer.utils.evaluation import evaluate_prompt


class EvaluatePromptsStage(BaseStage):
    """Evaluate prompts against test cases."""

    stage_name: Literal["quick_filter", "rigorous"]

    def __init__(self, stage_name: Literal["quick_filter", "rigorous"], *args, **kwargs):
        """
        Initialize evaluation stage.

        Args:
            stage_name: Name for the evaluation stage (e.g., "quick_filter", "rigorous")
            *args, **kwargs: Passed to BaseStage
        """
        super().__init__(*args, **kwargs)
        self.stage_name = stage_name

    @property
    def name(self) -> str:
        """Return the stage name."""
        return f"Evaluate ({self.stage_name.replace('_', ' ').title()})"

    async def _run_async(self, context: RunContext) -> RunContext:
        """
        Evaluate prompts against test cases in parallel with global concurrency control.

        Args:
            context: Run context with prompts and tests

        Returns:
            Updated context with prompts having updated scores
        """
        # Determine which prompts and tests to use
        if self.stage_name == "quick_filter":
            prompts = context.initial_prompts
            tests = context.quick_tests
        else:  # rigorous
            prompts = context.top_k_prompts
            tests = context.rigorous_tests

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
                self.storage,
                parallel=True,
                semaphore=semaphore,
            )
            for prompt in prompts
        ]

        scores = await asyncio.gather(*eval_tasks)

        for prompt, avg_score in zip(prompts, scores, strict=True):
            prompt.average_score = avg_score
            prompt.stage = self.stage_name
            self.storage.save_prompt(prompt)

        self._print_progress(
            f"Evaluation complete (scores: {[f'{p.average_score:.2f}' for p in prompts[:5]]}...)"
        )

        return context

    async def _run_sync(self, context: RunContext) -> RunContext:
        """
        Evaluate prompts against test cases sequentially (no concurrency).

        Args:
            context: Run context with prompts and tests

        Returns:
            Updated context with prompts having updated scores
        """
        # Determine which prompts and tests to use
        if self.stage_name == "quick_filter":
            prompts = context.initial_prompts
            tests = context.quick_tests
        else:  # rigorous
            prompts = context.top_k_prompts
            tests = context.rigorous_tests

        total_evaluations = len(prompts) * len(tests)
        self._print_progress(
            f"Evaluating {len(prompts)} prompts × {len(tests)} tests "
            f"({total_evaluations} total evaluations, sequential mode)..."
        )

        # No semaphore in sequential mode - everything runs one at a time
        for i, prompt in enumerate(prompts, 1):
            self._print_progress(f"  Evaluating prompt {i}/{len(prompts)}...")
            avg_score = await evaluate_prompt(
                prompt,
                tests,
                context.task_spec,
                self.config,
                self.model_client,
                self.storage,
                parallel=False,
                semaphore=None,
            )
            prompt.average_score = avg_score
            prompt.stage = self.stage_name
            self.storage.save_prompt(prompt)

        self._print_progress(
            f"Evaluation complete (scores: {[f'{p.average_score:.2f}' for p in prompts[:5]]}...)"
        )

        return context
