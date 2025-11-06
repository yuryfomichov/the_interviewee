"""Configuration for the ABC journaling prompt optimization example."""

from __future__ import annotations

from prompt_optimizer.config import LLMConfig, OptimizerConfig, TestDistribution
from prompt_optimizer.types import TaskSpec

ABC_SYSTEM_PROMPT = """You are an assistant in a smart journaling application called ABC. Your task is to help the user understand themselves, improve their mental health, answer their questions based on their notes in the journal, and so on. You will answer questions from the owner of the journal, be able to search through entries in the journal, analyze them, and make recommendations.

Follow these rules:
- communicate informally, as if you were a friend of the user
- do not answer in general terms, rely on the information given by the user in their notes
- if there is no direct answer to a question in the journal, try to give a recommendation based on your knowledge (mention that you couldn't find entry in notes)
- answer in the same language user wrote their notes
- always use absolute dates when you want to mention an event
- only switch to a structured STAR (Situation, Task, Action, Result) breakdown if the user explicitly asks for that format or clearly needs a step-by-step recap"""


def create_task_spec() -> TaskSpec:
    """Create the task specification for the ABC journaling assistant."""
    return TaskSpec(
        task_description=(
            "Assistant for the ABC smart journaling application that helps the user reflect on "
            "their notes, understand patterns, and receive tailored recommendations rooted in the "
            "journal content."
        ),
        behavioral_specs="""
Expected behavior:
- Communicate informally in a friendly, supportive tone
- Ground every answer in specific journal entries when possible
- Highlight patterns, insights, and recommendations drawn from the notes
- Use absolute dates (for example, March 18, 2024) when referencing events
- Respond in the same language as the provided journal entries
- Acknowledge when information is missing while still offering helpful suggestions
- Reserve STAR (Situation, Task, Action, Result) formatting for the rare cases when the user explicitly asks for a structured recap of past events; default to conversational summaries otherwise
        """,
        validation_rules=[
            "Never provide generic advice without referencing journal entries when they exist",
            "State explicitly when the relevant information is not present in the journal",
            "Maintain an informal, friendly tone throughout the response",
            "Preserve the user's language in replies",
            "Use absolute date formats when discussing events",
            "Only use STAR formatting when the user explicitly requests it or clearly needs a structured breakdown",
        ],
        current_prompt=ABC_SYSTEM_PROMPT,
    )


def create_optimizer_config(api_key: str) -> OptimizerConfig:
    """Create optimizer configuration tuned for the journaling assistant."""
    return OptimizerConfig(
        num_initial_prompts=3,
        quick_test_distribution=TestDistribution(
            core=1,
            edge=1,
            boundary=1,
            adversarial=0,
            consistency=0,
            format=0,
        ),
        top_k_advance=1,
        rigorous_test_distribution=TestDistribution(
            core=1,
            edge=1,
            boundary=1,
            adversarial=1,
            consistency=1,
            format=1,
        ),
        top_m_refine=1,
        max_iterations_per_track=2,
        convergence_threshold=0.02,
        early_stopping_patience=1,
        scoring_weights={
            "functionality": 0.35,
            "safety": 0.15,
            "consistency": 0.25,
            "edge_case_handling": 0.25,
        },
        generator_llm=LLMConfig(model="gpt-5-nano"),
        test_designer_llm=LLMConfig(model="gpt-5-nano"),
        evaluator_llm=LLMConfig(model="gpt-5-nano"),
        refiner_llm=LLMConfig(model="gpt-5-nano"),
        output_dir="prompt_optimizer/examples/abc_journaling/prompt_optimizer_data",
        parallel_execution=True,
        max_concurrent_evaluations=15,
        verbose=True,
        openai_api_key=api_key,
        task_spec=create_task_spec(),
    )
