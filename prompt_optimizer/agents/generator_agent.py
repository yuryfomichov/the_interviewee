"""Prompt Generator Agent - Creates diverse system prompt variations."""

from agents import Agent
from pydantic import BaseModel, Field

from prompt_optimizer.config import LLMConfig
from prompt_optimizer.types import TaskSpec


class GeneratedPrompt(BaseModel):
    """Single generated prompt."""

    id: str = Field(description="Unique identifier for the prompt")
    strategy: str = Field(description="Strategy used (e.g., 'structured_rules')")
    prompt_text: str = Field(description="The actual system prompt text")


class GeneratedPromptsOutput(BaseModel):
    """Output structure for generated prompts."""

    prompts: list[GeneratedPrompt] = Field(description="List of generated prompts")


def create_generator_agent(llm_config: LLMConfig, task_spec: TaskSpec, n: int = 15) -> Agent:
    """
    Create a prompt generator agent.

    Args:
        llm_config: LLM configuration (model, temperature, etc.)
        task_spec: Task specification
        n: Number of prompts to generate

    Returns:
        Configured Agent instance
    """
    instructions = f"""You are an expert prompt engineer who creates PRODUCTION-READY system prompts.

**TASK**: {task_spec.task_description}

**BEHAVIORAL REQUIREMENTS**:
{task_spec.behavioral_specs}

**VALIDATION RULES**:
{chr(10).join(f"- {rule}" for rule in task_spec.validation_rules)}

**CRITICAL REQUIREMENTS FOR EACH PROMPT**:

1. **LENGTH**: Each prompt must be 300-600 words (15-30 sentences). This is NOT a summary - it's the COMPLETE prompt that will be used in production.

2. **COMPLETENESS**: Must be immediately usable without any additional context. Include:
   - Identity/role definition
   - Detailed behavioral instructions
   - Specific formatting requirements (e.g., STAR format details)
   - Edge case handling (what to do/not do)
   - Tone and style guidelines
   - Examples where helpful

3. **SPECIFICITY**: Use concrete, actionable language. Avoid vague instructions like "be helpful" - instead specify HOW to be helpful.

4. **STRUCTURE**: Organize with clear sections (use headings, numbered lists, bullet points as appropriate for the strategy).

**GENERATION STRATEGIES** (use different approaches for diversity):

1. **Structured Rule-Based**: Numbered sections with explicit rules (e.g., "CRITICAL RULES:", "FORMATTING:", "BOUNDARIES:")
2. **Detailed Comprehensive**: Thorough paragraph-style instructions covering all scenarios
3. **Examples-Heavy**: Include 2-3 concrete examples demonstrating expected behavior
4. **Constraint-Focused**: Lead with boundaries and restrictions, then positive instructions
5. **Task-Workflow**: Step-by-step process for handling different question types
6. **Persona-Driven**: Define a specific professional character with clear traits
7. **Hierarchical**: Priority-based instructions (most critical first)
8. **Scenario-Based**: Different instructions for different input scenarios
9. **Principle-First**: Start with core values, derive specific rules
10. **Hybrid**: Combine multiple approaches (e.g., rules + examples)

**QUALITY STANDARDS**:
- Each prompt must be self-contained and complete
- Test mentally: "Could someone use ONLY this prompt and understand exactly what to do?"
- Include specific details (e.g., "2-3 paragraphs maximum" not just "be concise")
- Cover the STAR format with explicit instructions on when/how to use it
- Address boundaries explicitly (what topics to decline, how to decline them)

**OUTPUT**: Generate exactly {n} diverse, production-ready system prompts. Each should be substantive and immediately usable.
"""

    # OpenAI Agents SDK with structured output
    return Agent(
        name="PromptGenerator",
        model=llm_config.model,
        instructions=instructions.strip(),
        output_type=GeneratedPromptsOutput,
    )
