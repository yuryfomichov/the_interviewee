"""Refiner Agent - Iterative prompt improvement."""

from agents import Agent
from pydantic import BaseModel, Field

from prompt_optimizer.config import LLMConfig
from prompt_optimizer.types import TaskSpec


class RefinedPromptOutput(BaseModel):
    """Output structure for refined prompts."""

    improved_prompt: str = Field(description="The improved system prompt text")
    changes_made: str = Field(description="Brief description of improvements made")


def create_refiner_agent(
    llm_config: LLMConfig,
    task_spec: TaskSpec,
    current_prompt: str,
    weaknesses: str,
    failed_tests: list[str],
    iteration: int,
) -> Agent:
    """
    Create a refiner agent for improving a prompt based on identified weaknesses.

    Args:
        llm_config: LLM configuration
        task_spec: Task specification
        current_prompt: The prompt to improve
        weaknesses: Description of identified weak areas
        failed_tests: List of test cases that failed
        iteration: Current iteration number

    Returns:
        Configured Agent instance
    """
    failed_tests_str = "\n".join(f"- {test}" for test in failed_tests) if failed_tests else "None"

    instructions = f"""You are a prompt optimization specialist who surgically improves system prompts.

**TASK**: {task_spec.task_description}

**REQUIRED BEHAVIOR**:
{task_spec.behavioral_specs}

**VALIDATION RULES**:
{chr(10).join(f"- {rule}" for rule in task_spec.validation_rules)}

**CURRENT PROMPT** (Iteration {iteration}):
```
{current_prompt}
```

**IDENTIFIED WEAKNESSES**:
{weaknesses}

**FAILED TEST CASES**:
{failed_tests_str}

**YOUR JOB**:
Create an improved version of the system prompt that:

1. **Preserves Strengths**: Keep what works well (high-scoring aspects)
2. **Addresses Failures**: Fix specific failure modes from the test results
3. **Adds Clarifications**: Where confusion or ambiguity occurred
4. **Strengthens Boundaries**: Where rules were violated or misunderstood
5. **Maintains Conciseness**: Don't make it unnecessarily verbose

**IMPROVEMENT STRATEGIES**:
- Add explicit examples for areas where the AI failed
- Strengthen language around violated rules
- Add constraints or formatting instructions if needed
- Clarify tone/style requirements if inconsistent
- Reorder instructions to emphasize critical parts

Focus on targeted, surgical improvements. This is iteration {iteration}, so build on previous refinements.
"""

    # OpenAI Agents SDK with structured output
    return Agent(
        name="PromptRefiner",
        model=llm_config.model,
        instructions=instructions.strip(),
        output_type=RefinedPromptOutput,
    )
