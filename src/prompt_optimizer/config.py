"""Optimizer configuration for the AI interviewee prompt optimization.

This module defines the optimizer configuration with test distributions,
task specification, and model settings tailored for the interview assistant.

Adjust these values for faster testing during development:
- Use smaller numbers (e.g., 3 prompts, 2 quick tests, 5 rigorous tests)
- Runtime scales linearly with these numbers
"""

from prompt_optimizer.config import LLMConfig, OptimizerConfig, TestDistribution
from prompt_optimizer.types import TaskSpec


def create_task_spec() -> TaskSpec:
    """
    Create task specification for the AI interview assistant.

    Returns:
        TaskSpec describing behavioral expectations and guardrails.
    """
    return TaskSpec(
        task_description=(
            "AI interview assistant that answers questions based on career data "
            "using RAG (Retrieval-Augmented Generation)"
        ),
        behavioral_specs="""
Expected behavior:
- Answer behavioral questions using STAR format (Situation, Task, Action, Result)
- Only apply STAR format when questions ask about past experiences
- Cite specific experiences from retrieved career documents
- Maintain first-person professional voice
- Be concise and direct, avoid meta-commentary
- Use retrieved context as evidence for claims
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
        current_prompt="""### Role Definition
You are an AI interview assistant specializing in providing insightful responses to career-related inquiries, utilizing a retrieval-augmented generation approach.

### CRITICAL RULES
1. Behavioral Questions: Answer using the STAR format (Situation, Task, Action, Result) when prompted about past experiences.
2. Privacy Considerations: Do not share sensitive personal information.
3. Document Utilization: Base answers on context from retrieved career documents—avoid inventions or speculations.
4. Declining Topics: Politely refuse to engage in political or non-professional topics.
5. Contextual Boundaries: Maintain focus strictly on career and professional topics.

### FORMATTING
- Conciseness: Responses should be clear and direct, typically comprising 2-3 paragraphs.
- Voice/Tone: Use a first-person professional voice, maintaining a helpful and respectful tone.

### EXAMPLES
- If asked about leadership, cite a specific leadership experience from the retrieved documents using STAR format.
- For questions outside your scope (e.g., political inquiries), respond with: "I focus on career-related topics and am unable to discuss that area."

### EDGE CASES
- When context is unavailable, express politely that you need more information.
- Do not fabricate details; rely solely on available data.""",
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
        # Stage 1: Quick Filter
        num_initial_prompts=5,
        quick_test_distribution=TestDistribution(
            core=1,
            edge=1,
            boundary=1,
            adversarial=0,
            consistency=1,
            format=0,
        ),
        top_k_advance=2,
        # Stage 2: Rigorous Testing
        rigorous_test_distribution=TestDistribution(
            core=2,
            edge=2,
            boundary=2,
            adversarial=1,
            consistency=1,
            format=1,
        ),
        top_m_refine=1,
        # Stage 3: Refinement
        max_iterations_per_track=2,
        # LLM models for each agent
        generator_llm=LLMConfig(model="gpt-4o"),
        test_designer_llm=LLMConfig(model="gpt-4o"),
        evaluator_llm=LLMConfig(model="gpt-4o"),
        refiner_llm=LLMConfig(model="gpt-4o"),
        # Output
        storage_path="prompt_optimizer/data/optimizer.db",
        verbose=True,
        # API Configuration
        openai_api_key=api_key,
        # Task Spec
        task_spec=create_task_spec(),
    )
