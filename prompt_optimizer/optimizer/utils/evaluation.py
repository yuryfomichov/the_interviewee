"""Shared evaluation logic for testing prompts."""

from agents import Runner

from prompt_optimizer.agents.evaluator_agent import create_evaluator_agent
from prompt_optimizer.config import OptimizerConfig
from prompt_optimizer.connectors import BaseConnector
from prompt_optimizer.optimizer.utils.model_tester import test_target_model
from prompt_optimizer.optimizer.utils.score_calculator import aggregate_prompt_score
from prompt_optimizer.storage import Storage
from prompt_optimizer.types import (
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
    storage: Storage,
) -> float:
    """
    Evaluate a prompt against test cases and return average score.

    Args:
        prompt: Prompt candidate to evaluate
        test_cases: Test cases to run
        task_spec: Task specification
        config: Optimizer configuration
        model_client: Connector for the target model
        storage: Storage instance for saving results

    Returns:
        Average score across all test cases
    """
    evaluations = []

    for test in test_cases:
        # Get model response
        response = test_target_model(prompt.prompt_text, test.input_message, model_client)

        # Score with LLM judge
        evaluator = create_evaluator_agent(config.evaluator_llm, task_spec, test)
        eval_result = await Runner.run(
            evaluator,
            f"Score this response:\n\n{response}\n\nProvide scores in JSON format.",
        )

        # Parse evaluation
        eval_data = _parse_evaluation(eval_result.final_output)
        evaluation = EvaluationScore.calculate_overall(
            functionality=eval_data["functionality"],
            safety=eval_data["safety"],
            consistency=eval_data["consistency"],
            edge_case_handling=eval_data["edge_case_handling"],
            reasoning=eval_data["reasoning"],
            weights=config.scoring_weights,
        )

        # Save to storage
        test_result = TestResult(
            test_case_id=test.id,
            prompt_id=prompt.id,
            model_response=response,
            evaluation=evaluation,
        )
        storage.save_evaluation(test_result)
        evaluations.append(evaluation)

    return aggregate_prompt_score(evaluations)


def _parse_evaluation(agent_output) -> dict:
    """
    Parse agent evaluation output.

    Args:
        agent_output: EvaluationOutput object from the agent

    Returns:
        Dict with evaluation scores
    """
    # Agent returns a Pydantic object (EvaluationOutput)
    return agent_output.model_dump()
