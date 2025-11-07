"""Fake agent responses for testing - replaces OpenAI Agents SDK calls."""

import hashlib
import random
from typing import Any
from unittest.mock import AsyncMock

from prompt_optimizer.agents.evaluator_agent import EvaluationOutput
from prompt_optimizer.agents.prompt_generator_agent import GeneratedPrompt, GeneratedPromptsOutput
from prompt_optimizer.agents.refiner_agent import RefinedPromptOutput
from prompt_optimizer.agents.test_designer_agent import TestCasesOutput
from prompt_optimizer.schemas import TestCase


class FakeRunnerResult:
    """Mimics the result structure from agents.Runner.run()."""

    def __init__(self, final_output: Any):
        self.final_output = final_output


async def fake_runner_run(agent, task_description: str) -> FakeRunnerResult:
    """Fake implementation of agents.Runner.run()."""
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
    """Generate N simple prompts - just need valid count."""
    instructions = agent.instructions.lower()

    # Determine count from instructions
    if "exactly 3 diverse" in instructions:
        n = 3
    elif "exactly 15 diverse" in instructions:
        n = 15
    else:
        n = 15

    prompts = [
        GeneratedPrompt(id=f"p{i}", strategy=f"s{i}", prompt_text=f"prompt{i}")
        for i in range(n)
    ]
    return GeneratedPromptsOutput(prompts=prompts)


def create_fake_test_designer_response(agent) -> TestCasesOutput:
    """Generate test cases - just need valid count and categories."""
    instructions = agent.instructions

    # Pick distribution based on total count in instructions
    if "Create EXACTLY 2 evaluation tests" in instructions:
        dist = {"core": 1, "edge": 1}
    elif "Create EXACTLY 3 evaluation tests" in instructions:
        dist = {"core": 2, "edge": 1}
    elif "Create EXACTLY 7 evaluation tests" in instructions:
        dist = {"core": 2, "edge": 2, "boundary": 1, "adversarial": 1, "consistency": 1}
    elif "Create EXACTLY 50 evaluation tests" in instructions:
        dist = {"core": 20, "edge": 10, "boundary": 10, "adversarial": 5, "consistency": 3, "format": 2}
    else:
        dist = {"core": 20, "edge": 10, "boundary": 10, "adversarial": 5, "consistency": 3, "format": 2}

    test_cases = []
    test_id = 0
    for category, count in dist.items():
        for i in range(count):
            test_cases.append(
                TestCase(
                    id=f"t{test_id}",
                    input_message="test",
                    expected_behavior="expected",
                    category=category
                )
            )
            test_id += 1
    return TestCasesOutput(test_cases=test_cases)


def create_fake_evaluator_response(agent) -> EvaluationOutput:
    """Generate random scores - needed for selection tests."""
    seed = int(hashlib.md5(agent.instructions.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    return EvaluationOutput(
        functionality=rng.randint(6, 10),
        safety=rng.randint(6, 10),
        consistency=rng.randint(5, 10),
        edge_case_handling=rng.randint(5, 9),
        reasoning="ok",
    )


def create_fake_refiner_response(agent) -> RefinedPromptOutput:
    """Create simple refinement - just need to return something."""
    instructions = agent.instructions

    # Extract current prompt if present
    if "CURRENT PROMPT:" in instructions:
        try:
            current_prompt = instructions.split("CURRENT PROMPT:")[1].split("WEAKNESSES:")[0].strip()
        except IndexError:
            current_prompt = "prompt"
    else:
        current_prompt = "prompt"

    return RefinedPromptOutput(
        improved_prompt=current_prompt + "\nrefined",
        changes_made="refined",
    )


def setup_fake_agents(monkeypatch) -> None:
    """Set up fake agents by patching agents.Runner.run."""
    from agents import Runner
    async_mock = AsyncMock(side_effect=fake_runner_run)
    monkeypatch.setattr(Runner, "run", async_mock)
