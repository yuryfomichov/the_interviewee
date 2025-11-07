"""Tools for calculating evaluation scores."""

import logging

from prompt_optimizer.schemas import EvaluationScore

logger = logging.getLogger(__name__)


def aggregate_prompt_score(evaluations: list[EvaluationScore]) -> float:
    """
    Aggregate scores across multiple evaluations.

    Args:
        evaluations: List of evaluation scores

    Returns:
        Average overall score
    """
    if not evaluations:
        return 0.0

    avg_score = sum(e.overall for e in evaluations) / len(evaluations)
    logger.info(f"Aggregated {len(evaluations)} evaluations: avg score = {avg_score:.2f}")
    return avg_score
