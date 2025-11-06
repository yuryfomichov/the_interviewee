"""
Prompt Optimizer - AI-powered system prompt optimization using multi-agent orchestration.

This package provides tools to iteratively generate, test, and refine system prompts
to achieve optimal AI behavior through a 3-stage pipeline:
1. Quick Filter: Generate 15 prompts, test with 7 cases, select top 5
2. Rigorous Testing: Test top 5 with 50 cases, select top 3
3. Parallel Refinement: Refine top 3 in parallel tracks, select champion

Public API:
- BaseConnector: Abstract base class for implementing custom connectors
- OpenAIConnector: Built-in connector for OpenAI models
- OptimizerConfig: Configuration for the optimization pipeline
- OptimizationRunner: Main runner interface for executing optimization
"""

from prompt_optimizer.config import OptimizerConfig
from prompt_optimizer.connectors import BaseConnector, OpenAIConnector
from prompt_optimizer.runner import OptimizationRunner

__all__ = [
    "BaseConnector",
    "OpenAIConnector",
    "OptimizerConfig",
    "OptimizationRunner",
]
