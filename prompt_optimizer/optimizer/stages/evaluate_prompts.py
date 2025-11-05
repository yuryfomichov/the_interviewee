"""Evaluate prompts stage: Run prompts against test cases."""

import asyncio

from prompt_optimizer.optimizer.base_stage import BaseStage
from prompt_optimizer.optimizer.context import RunContext
from prompt_optimizer.optimizer.utils.evaluation import evaluate_prompt


class EvaluatePromptsStage(BaseStage):
    """Evaluate prompts against test cases."""

    def __init__(self, stage_name: str, *args, **kwargs):
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

    async def run(self, context: RunContext) -> RunContext:
        """
        Evaluate prompts against test cases.

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

        self._print_progress(
            f"Evaluating {len(prompts)} prompts × {len(tests)} tests (parallel)..."
        )

        eval_tasks = [
            evaluate_prompt(
                prompt, tests, context.task_spec, self.config, self.model_client, self.storage
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
