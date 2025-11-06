"""Generate tests stage: Create test cases for evaluation."""

from agents import Runner

from prompt_optimizer.agents.test_designer_agent import (
    TestCasesOutput,
    create_test_designer_agent,
)
from prompt_optimizer.optimizer.base_stage import BaseStage
from prompt_optimizer.optimizer.context import RunContext
from prompt_optimizer.types import TestCase


class GenerateTestsStage(BaseStage):
    """Generate test cases for a specific testing stage."""

    def __init__(self, test_stage: str, *args, **kwargs):
        """
        Initialize test generation stage.

        Args:
            test_stage: "quick" or "rigorous"
            *args, **kwargs: Passed to BaseStage
        """
        super().__init__(*args, **kwargs)
        self.test_stage = test_stage

    @property
    def name(self) -> str:
        """Return the stage name."""
        return f"Generate {self.test_stage.capitalize()} Tests"

    async def _run_async(self, context: RunContext) -> RunContext:
        """
        Generate test cases (async mode).

        Args:
            context: Run context with task_spec

        Returns:
            Updated context with quick_tests or rigorous_tests populated
        """
        num_tests = (
            self.config.num_quick_tests
            if self.test_stage == "quick"
            else self.config.num_rigorous_tests
        )
        distribution = (
            self.config.quick_test_distribution
            if self.test_stage == "quick"
            else self.config.rigorous_test_distribution
        )

        self._print_progress(f"Designing {num_tests} {self.test_stage} tests...")

        test_designer = create_test_designer_agent(
            self.config.test_designer_llm,
            context.task_spec,
            distribution,
            stage=self.test_stage,
        )
        test_result = await Runner.run(
            test_designer,
            f"Create {self.test_stage} evaluation tests",
        )
        tests = self._parse_test_cases(test_result.final_output)

        # Save tests to storage
        for test in tests:
            self.storage.save_test_case(test, self.test_stage)

        self._print_progress(f"Generated {len(tests)} {self.test_stage} test cases")

        # Update context
        if self.test_stage == "quick":
            context.quick_tests = tests
        else:
            context.rigorous_tests = tests

        return context

    async def _run_sync(self, context: RunContext) -> RunContext:
        """
        Generate test cases (sync mode - same as async for this stage).

        Args:
            context: Run context with task_spec

        Returns:
            Updated context with quick_tests or rigorous_tests populated
        """
        # This stage doesn't benefit from parallel execution since it's a single agent call
        return await self._run_async(context)

    def _parse_test_cases(self, agent_output: TestCasesOutput) -> list[TestCase]:
        """Parse agent output into TestCase objects."""
        return agent_output.test_cases
