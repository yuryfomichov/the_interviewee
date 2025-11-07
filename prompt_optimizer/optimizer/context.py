"""Minimal run context for passing metadata between optimization stages."""

from typing import Any

from pydantic import BaseModel, Field, PrivateAttr
from sqlalchemy.orm import Session

from prompt_optimizer.storage.repositories import (
    EvaluationRepository,
    PromptRepository,
    RunRepository,
    TestCaseRepository,
)
from prompt_optimizer.schemas import TaskSpec


class RunContext(BaseModel):
    """
    Minimal context object passed between optimization stages.

    Instead of carrying all data in memory, this context only stores:
    - Identification (run_id)
    - Configuration (task_spec)
    - Metadata (timing, output paths)
    - Database session (for querying data as needed)

    Stages query data from the database on-demand rather than passing it through context.
    """

    # Run identification
    run_id: int = Field(description="Optimization run ID")

    # Task specification (small, needed everywhere)
    task_spec: TaskSpec = Field(description="Task specification for optimization")

    # Execution metadata
    start_time: float = Field(description="When optimization started (unix timestamp)")
    output_dir: str = Field(description="Directory for saving reports and artifacts")

    # Database session (private attribute, not a field)
    _session: Any = PrivateAttr(default=None)
    _optimization_result: Any = PrivateAttr(default=None)

    model_config = {"arbitrary_types_allowed": True}

    # === Repository access helpers ===

    @property
    def prompt_repo(self) -> PromptRepository:
        """Get prompt repository."""
        if self._session is None:
            raise RuntimeError("Database session not set on context")
        return PromptRepository(self._session)

    @property
    def test_repo(self) -> TestCaseRepository:
        """Get test case repository."""
        if self._session is None:
            raise RuntimeError("Database session not set on context")
        return TestCaseRepository(self._session)

    @property
    def eval_repo(self) -> EvaluationRepository:
        """Get evaluation repository."""
        if self._session is None:
            raise RuntimeError("Database session not set on context")
        return EvaluationRepository(self._session)

    @property
    def run_repo(self) -> RunRepository:
        """Get run repository."""
        if self._session is None:
            raise RuntimeError("Database session not set on context")
        return RunRepository(self._session)

    def set_session(self, session: Session) -> None:
        """
        Set the database session on this context.

        Args:
            session: SQLAlchemy session instance
        """
        self._session = session
