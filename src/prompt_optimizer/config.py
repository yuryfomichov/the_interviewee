"""Optimizer configuration for the AI interviewee prompt optimization.

This module defines the optimizer configuration with test distributions,
task specification, and model settings tailored for the interview assistant.

Adjust these values for faster testing during development:
- Use smaller numbers (e.g., 3 prompts, 2 quick tests, 5 rigorous tests)
- Runtime scales linearly with these numbers
"""

from prompt_optimizer.config import LLMConfig, OptimizerConfig, TestDistribution
from prompt_optimizer.types import TaskSpec
from prompts import get_default_system_prompt


def create_task_spec() -> TaskSpec:
    """
    Create task specification for the AI interview assistant.

    Returns:
        TaskSpec describing behavioral expectations and guardrails.
    """
    return TaskSpec(
        task_description=(
            "AI interviewee specializing in providing insightful responses to career-related inquiries about a specific person. You have documents about the person and utilize a retrieval-augmented generation approach to retrieve relevant information."
        ),
        behavioral_specs="""
Expected behavior:
- Answer behavioral questions using STAR format (Situation, Task, Action, Result)
- Only apply STAR format when questions ask about past experiences
- When given multiple prompts in one request, answer each one with consistent structure, tone, and format
- Maintain comparable brevity across all answers in multi-prompt requests
- Cite specific experiences from retrieved career documents
- Maintain first-person professional voice
- Be concise and direct, avoid meta-commentary (typically 2-4 sentences per answer)
- Use retrieved context as evidence for claims
- If evidence is missing, explicitly state that and ask for clarification
- Never invent experiences or fabricate details
- Professional and helpful tone
        """,
        validation_rules=[
            "Decline political questions politely",
            "Stay within career/professional context",
            "Use retrieved documents as sources - don't invent experiences",
            "No speculation or made-up information",
            "Respect privacy - don't share sensitive personal details",
            "Follow STAR format only for behavioral/experience questions",
        ],
        current_prompt=get_default_system_prompt(user_name="The Interviewee AI"),
    )


def create_optimizer_config(api_key: str) -> OptimizerConfig:
    """
    Create optimizer configuration for the interview assistant.

    Args:
        api_key: OpenAI API key for optimizer agents

    Returns:
        Configured OptimizerConfig instance with hardcoded settings
    """
    return OptimizerConfig(
        # Stage: Quick Filter
        num_initial_prompts=3,
        quick_test_distribution=TestDistribution(
            core=1,
            edge=1,
            boundary=1,
            adversarial=1,
            consistency=1,
            format=1,
        ),
        top_k_advance=1,
        # Stage: Rigorous Testing
        rigorous_test_distribution=TestDistribution(
            core=2,
            edge=2,
            boundary=2,
            adversarial=2,
            consistency=2,
            format=2,
        ),
        top_m_refine=1,
        # Stage: Refinement
        max_iterations_per_track=3,
        # LLM models for each agent
        generator_llm=LLMConfig(model="gpt-5-nano"),
        test_designer_llm=LLMConfig(model="gpt-5-nano"),
        evaluator_llm=LLMConfig(model="gpt-5-nano"),
        refiner_llm=LLMConfig(model="gpt-5-nano"),
        verbose=True,
        # API Configuration
        openai_api_key=api_key,
        # Task Spec
        task_spec=create_task_spec(),
        parallel_execution=True,
        max_concurrent_evaluations=8,
    )
