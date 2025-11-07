"""Test Designer Agent - Creates comprehensive test cases."""

from agents import Agent
from pydantic import BaseModel, Field

from prompt_optimizer.config import LLMConfig, TestDistribution
from prompt_optimizer.schemas import TaskSpec, TestCase


class TestCasesOutput(BaseModel):
    """Output structure for test cases."""

    test_cases: list[TestCase] = Field(description="List of test cases")


def create_test_designer_agent(
    llm_config: LLMConfig,
    task_spec: TaskSpec,
    test_distribution: TestDistribution,
    stage: str = "quick",
) -> Agent:
    """
    Create a test designer agent.

    Args:
        llm_config: LLM configuration
        task_spec: Task specification
        test_distribution: Distribution of test categories
        stage: "quick" or "rigorous" (for prompt emphasis only)

    Returns:
        Configured Agent instance
    """
    # Get descriptions from the TestDistribution Pydantic model
    category_descriptions = {
        field_name: field_info.description
        for field_name, field_info in TestDistribution.model_fields.items()
    }

    # Build distribution items with descriptions
    dist_items = [
        f"- **{cat}**: EXACTLY {count} tests ({category_descriptions[cat]})"
        for cat, count in test_distribution.to_dict().items()
        if count > 0  # Only include categories with tests
    ]
    dist_str = "\n".join(dist_items)

    # Different emphasis based on stage
    stage_emphasis = (
        "Focus on high-signal tests that quickly reveal prompt quality."
        if stage == "quick"
        else "Be creative and thorough within each category. Think like a QA engineer trying to find weaknesses."
    )

    focus = f"""Create EXACTLY {test_distribution.total} evaluation tests following this EXACT distribution:
{dist_str}

CRITICAL: You MUST create the exact number specified for each category. Do not deviate from these counts.

{stage_emphasis}"""

    instructions = f"""You are a meticulous QA engineer who designs comprehensive test cases.

**TASK BEING TESTED**: {task_spec.task_description}

**EXPECTED BEHAVIOR**:
{task_spec.behavioral_specs}

**VALIDATION RULES**:
{chr(10).join(f"- {rule}" for rule in task_spec.validation_rules)}

{focus}

**Categories**: core, edge, boundary, adversarial, consistency, format

**USER MESSAGE STYLE REQUIREMENTS**:
- Phrase each input_message as a natural request or instruction consistent with the task scenario. Do not prefix requests with meta labels (e.g., "Behavioral question:", "Format test:", "Coding task:") unless the domain normally requires those markers.
- Reference supporting resources only when that interaction is part of the task expectations, and describe them using the terminology the scenario would naturally use (e.g., files for coding agents, brief summaries for interview coaching).
- When you mention prior context, do so conversationally rather than listing resource names unless explicit identifiers are normal in that domain.

Make tests specific, actionable, and diverse. Each test should reveal something important about prompt quality.
"""

    # OpenAI Agents SDK with structured output
    return Agent(
        name="TestDesigner",
        model=llm_config.model,
        instructions=instructions.strip(),
        output_type=TestCasesOutput,
    )
