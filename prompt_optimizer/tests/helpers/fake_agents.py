"""Fake agent responses for testing - replaces OpenAI Agents SDK calls."""

import hashlib
import random
import uuid
from typing import Any
from unittest.mock import AsyncMock

from prompt_optimizer.agents.evaluator_agent import EvaluationOutput
from prompt_optimizer.agents.prompt_generator_agent import GeneratedPrompt, GeneratedPromptsOutput
from prompt_optimizer.agents.refiner_agent import RefinedPromptOutput
from prompt_optimizer.agents.test_designer_agent import TestCasesOutput
from prompt_optimizer.schemas import TestCase

# Test configuration constants - used to generate consistent test data
DEFAULT_NUM_PROMPTS = 15

TEST_DISTRIBUTIONS = {
    "minimal_quick": {"core": 1, "edge": 1, "boundary": 0, "adversarial": 0, "consistency": 0, "format": 0},
    "minimal_rigorous": {"core": 2, "edge": 1, "boundary": 0, "adversarial": 0, "consistency": 0, "format": 0},
    "realistic_quick": {"core": 2, "edge": 2, "boundary": 1, "adversarial": 1, "consistency": 1, "format": 0},
    "realistic_rigorous": {"core": 20, "edge": 10, "boundary": 10, "adversarial": 5, "consistency": 3, "format": 2},
}

DEFAULT_SCORING_WEIGHTS = {
    "functionality": 0.4,
    "safety": 0.3,
    "consistency": 0.2,
    "edge_case_handling": 0.1
}


class FakeRunnerResult:
    """Mimics the result structure from agents.Runner.run()."""

    def __init__(self, final_output: Any):
        self.final_output = final_output


async def fake_runner_run(agent, task_description: str) -> FakeRunnerResult:
    """
    Fake implementation of agents.Runner.run().
    Returns appropriate fake responses based on agent type.
    """
    agent_name = agent.name if hasattr(agent, "name") else "Unknown"

    if agent_name == "PromptGenerator":
        return FakeRunnerResult(create_fake_generator_response(agent))
    elif agent_name == "TestDesigner":
        return FakeRunnerResult(create_fake_test_designer_response(agent))
    elif agent_name == "Evaluator":
        return FakeRunnerResult(create_fake_evaluator_response(agent))
    elif agent_name in ("Refiner", "PromptRefiner"):
        return FakeRunnerResult(create_fake_refiner_response(agent))
    else:
        raise ValueError(f"Unknown agent type: {agent_name}")


def create_fake_generator_response(agent) -> GeneratedPromptsOutput:
    """Generate N simple prompts with different IDs."""
    # Simple pattern: look for "exactly N diverse" in instructions
    instructions = agent.instructions.lower()
    if "exactly 3 diverse" in instructions:
        n = 3
    elif "exactly 15 diverse" in instructions:
        n = 15
    else:
        n = DEFAULT_NUM_PROMPTS

    strategies = ["strategy_a", "strategy_b", "strategy_c", "strategy_d", "strategy_e"]
    prompts = []

    for i in range(n):
        strategy = strategies[i % len(strategies)]
        prompts.append(
            GeneratedPrompt(
                id=f"prompt_{uuid.uuid4().hex[:8]}",
                strategy=strategy,
                prompt_text=f"Test system prompt {i+1} using {strategy}"
            )
        )

    return GeneratedPromptsOutput(prompts=prompts)


def create_fake_test_designer_response(agent) -> TestCasesOutput:
    """Generate test cases based on distribution constants."""
    instructions = agent.instructions

    # Simple pattern matching to pick distribution
    if "EXACTLY 2 evaluation tests" in instructions:
        distribution = TEST_DISTRIBUTIONS["minimal_quick"]
    elif "EXACTLY 3 evaluation tests" in instructions:
        distribution = TEST_DISTRIBUTIONS["minimal_rigorous"]
    elif "EXACTLY 7 evaluation tests" in instructions:
        distribution = TEST_DISTRIBUTIONS["realistic_quick"]
    elif "EXACTLY 50 evaluation tests" in instructions:
        distribution = TEST_DISTRIBUTIONS["realistic_rigorous"]
    else:
        distribution = TEST_DISTRIBUTIONS["realistic_rigorous"]

    test_cases = []
    for category, count in distribution.items():
        for i in range(count):
            test_cases.append(
                TestCase(
                    id=f"test_{category}_{i}_{uuid.uuid4().hex[:6]}",
                    input_message=f"{category.title()} test #{i+1}",
                    expected_behavior=f"Should handle {category} correctly.",
                    category=category
                )
            )

    return TestCasesOutput(test_cases=test_cases)


def create_fake_evaluator_response(agent) -> EvaluationOutput:
    """Generate random scores for evaluation."""
    instructions = agent.instructions
    seed = int(hashlib.md5(instructions.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    functionality = rng.randint(6, 10)
    safety = rng.randint(6, 10)
    consistency = rng.randint(5, 10)
    edge_case_handling = rng.randint(5, 9)

    reasoning = f"Functionality: {functionality}/10, Safety: {safety}/10, Consistency: {consistency}/10, Edge cases: {edge_case_handling}/10"

    return EvaluationOutput(
        functionality=functionality,
        safety=safety,
        consistency=consistency,
        edge_case_handling=edge_case_handling,
        reasoning=reasoning,
    )


def create_fake_refiner_response(agent) -> RefinedPromptOutput:
    """Create simple refinement by appending suffix."""
    instructions = agent.instructions

    # Extract current prompt from instructions
    if "CURRENT PROMPT:" in instructions:
        try:
            current_prompt = instructions.split("CURRENT PROMPT:")[1].split("WEAKNESSES:")[0].strip()
        except IndexError:
            current_prompt = "Default test prompt"
    else:
        current_prompt = "Default test prompt"

    improved_prompt = current_prompt + "\n[Refined]"

    return RefinedPromptOutput(
        improved_prompt=improved_prompt,
        changes_made="Applied refinements.",
    )


def setup_fake_agents(monkeypatch) -> None:
    """
    Set up fake agents by patching agents.Runner.run.
    Replaces real agent calls with fast, deterministic fake responses.
    """
    from agents import Runner

    async_mock = AsyncMock(side_effect=fake_runner_run)
    monkeypatch.setattr(Runner, "run", async_mock)
