"""Shared evaluation logic for testing prompts."""

import asyncio

from agents import Runner

from prompt_optimizer.agents.evaluator_agent import EvaluationOutput, create_evaluator_agent
from prompt_optimizer.config import OptimizerConfig
from prompt_optimizer.connectors import BaseConnector
from prompt_optimizer.optimizer.context import RunContext
from prompt_optimizer.optimizer.utils.model_tester import test_target_model
from prompt_optimizer.optimizer.utils.score_calculator import aggregate_prompt_score
from prompt_optimizer.storage import EvaluationConverter
from prompt_optimizer.schemas import (
    EvaluationScore,
    PromptCandidate,
    TaskSpec,
    TestCase,
    TestResult,
)


async def evaluate_prompt(
    prompt: PromptCandidate,
    test_cases: list[TestCase],
    task_spec: TaskSpec,
    config: OptimizerConfig,
    model_client: BaseConnector,
    context: RunContext,
    parallel: bool = True,
    semaphore: asyncio.Semaphore | None = None,
) -> float:
    """
    Evaluate a single prompt against test cases and return average score.

    Args:
        prompt: Prompt candidate to evaluate
        test_cases: Test cases to run
        task_spec: Task specification
        config: Optimizer configuration
        model_client: Connector for the target model
        context: Run context for database access
        parallel: Whether to run evaluations in parallel (default: True)
        semaphore: Optional shared semaphore for global concurrency control.
                   If None and parallel=True, creates a local semaphore.

    Returns:
        Average score across all test cases
    """

    async def evaluate_single_test(test: TestCase) -> EvaluationScore:
        """Evaluate a single test case with optional concurrency control."""
        # Acquire semaphore if provided
        if semaphore:
            async with semaphore:
                return await _evaluate_test_impl(test)
        else:
            return await _evaluate_test_impl(test)

    async def _evaluate_test_impl(test: TestCase) -> EvaluationScore:
        """Implementation of single test evaluation."""
        # Get model response
        response = await test_target_model(prompt.prompt_text, test.input_message, model_client)

        # Score with LLM judge
        evaluator = create_evaluator_agent(config.evaluator_llm, task_spec, test)
        eval_result = await Runner.run(
            evaluator,
            f"Score this response:\n\n{response}\n\nProvide scores in JSON format.",
        )

        # Parse evaluation (final_output is EvaluationOutput due to agent's output_type)
        eval_output: EvaluationOutput = eval_result.final_output  # type: ignore[assignment]
        evaluation = EvaluationScore.calculate_overall(
            functionality=eval_output.functionality,
            safety=eval_output.safety,
            consistency=eval_output.consistency,
            edge_case_handling=eval_output.edge_case_handling,
            reasoning=eval_output.reasoning,
            weights=config.scoring_weights,
        )

        # Save evaluation to database
        test_result = TestResult(
            test_case_id=test.id,
            prompt_id=prompt.id,
            model_response=response,
            evaluation=evaluation,
        )
        db_evaluation = EvaluationConverter.to_db(test_result, context.run_id)
        context.eval_repo.save(db_evaluation)
        return evaluation

    # Run evaluations in parallel or sequentially based on config
    if parallel:
        evaluations = await asyncio.gather(*[evaluate_single_test(test) for test in test_cases])
    else:
        evaluations = []
        for test in test_cases:
            evaluation = await evaluate_single_test(test)
            evaluations.append(evaluation)

    return aggregate_prompt_score(evaluations)
