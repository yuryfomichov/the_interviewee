"""Repository for OptimizationRun data access."""

from datetime import datetime

from sqlalchemy.orm import Session

from prompt_optimizer.storage.models import OptimizationRun


class RunRepository:
    """Data access layer for optimization runs."""

    def __init__(self, session: Session):
        """Initialize repository with database session."""
        self.session = session

    def create(self, task_description: str) -> OptimizationRun:
        """
        Create a new optimization run.

        Args:
            task_description: Description of the optimization task

        Returns:
            Created run instance with ID
        """
        run = OptimizationRun(
            task_description=task_description,
            started_at=datetime.now(),
            status="running",
        )
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)  # Get the auto-generated ID
        return run

    def get_by_id(self, run_id: int) -> OptimizationRun | None:
        """
        Get run by ID.

        Args:
            run_id: Run ID

        Returns:
            OptimizationRun instance or None
        """
        return self.session.query(OptimizationRun).filter(OptimizationRun.id == run_id).first()

    def complete(
        self,
        run_id: int,
        champion_prompt_id: str,
        total_tests: int,
        total_time: float,
    ) -> None:
        """
        Mark a run as completed.

        Args:
            run_id: Run ID
            champion_prompt_id: ID of the winning prompt
            total_tests: Total number of tests executed
            total_time: Total execution time in seconds
        """
        run = self.get_by_id(run_id)
        if run:
            run.completed_at = datetime.now()
            run.champion_prompt_id = champion_prompt_id
            run.total_tests_run = total_tests
            run.total_time_seconds = total_time
            run.status = "completed"
            self.session.commit()

    def get_all(self, limit: int = 100) -> list[OptimizationRun]:
        """
        Get all runs ordered by most recent first.

        Args:
            limit: Maximum number of runs to return

        Returns:
            List of optimization runs
        """
        return (
            self.session.query(OptimizationRun)
            .order_by(OptimizationRun.started_at.desc())
            .limit(limit)
            .all()
        )

    def get_completed_runs(self, limit: int = 100) -> list[OptimizationRun]:
        """
        Get completed runs only.

        Args:
            limit: Maximum number of runs to return

        Returns:
            List of completed optimization runs
        """
        return (
            self.session.query(OptimizationRun)
            .filter(OptimizationRun.status == "completed")
            .order_by(OptimizationRun.started_at.desc())
            .limit(limit)
            .all()
        )
