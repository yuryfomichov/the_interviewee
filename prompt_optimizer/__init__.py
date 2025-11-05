"""
Prompt Optimizer - AI-powered system prompt optimization using multi-agent orchestration.

This package provides tools to iteratively generate, test, and refine system prompts
to achieve optimal AI behavior through a 3-stage pipeline:
1. Quick Filter: Generate 15 prompts, test with 7 cases, select top 5
2. Rigorous Testing: Test top 5 with 50 cases, select top 3
3. Parallel Refinement: Refine top 3 in parallel tracks, select champion
"""

from prompt_optimizer.config import OptimizerConfig, TestDistribution
from prompt_optimizer.model_clients import FunctionModelClient, ModelClient
from prompt_optimizer.optimizer import PromptOptimizer
from prompt_optimizer.types import (
    EvaluationScore,
    OptimizationResult,
    PromptCandidate,
    TaskSpec,
    TestCase,
)

__all__ = [
    "EvaluationScore",
    "OptimizationResult",
    "PromptCandidate",
    "TaskSpec",
    "TestCase",
    "OptimizerConfig",
    "TestDistribution",
    "FunctionModelClient",
    "ModelClient",
    "PromptOptimizer",
]
