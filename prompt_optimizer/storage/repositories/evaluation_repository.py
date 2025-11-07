"""Repository for Evaluation data access."""

from sqlalchemy.orm import Session, joinedload

from prompt_optimizer.storage.models import Evaluation


class EvaluationRepository:
    """Data access layer for evaluations."""

    def __init__(self, session: Session):
        """Initialize repository with database session."""
        self.session = session

    def save(self, evaluation: Evaluation) -> Evaluation:
        """
        Save an evaluation result.

        Args:
            evaluation: Evaluation instance to save

        Returns:
            Saved evaluation instance
        """
        self.session.add(evaluation)
        self.session.commit()
        return evaluation

    def get_by_id(self, evaluation_id: int) -> Evaluation | None:
        """
        Get evaluation by ID.

        Args:
            evaluation_id: Evaluation ID

        Returns:
            Evaluation instance or None
        """
        return self.session.query(Evaluation).filter(Evaluation.id == evaluation_id).first()

    def get_by_prompt(self, prompt_id: str) -> list[Evaluation]:
        """
        Get all evaluations for a specific prompt.

        Args:
            prompt_id: Prompt ID

        Returns:
            List of evaluations ordered by timestamp descending
        """
        return (
            self.session.query(Evaluation)
            .filter(Evaluation.prompt_id == prompt_id)
            .order_by(Evaluation.timestamp.desc())
            .all()
        )

    def get_by_prompt_with_tests(self, prompt_id: str) -> list[Evaluation]:
        """
        Get all evaluations for a prompt with test cases eagerly loaded.

        Args:
            prompt_id: Prompt ID

        Returns:
            List of evaluations with test_case relationship loaded
        """
        return (
            self.session.query(Evaluation)
            .filter(Evaluation.prompt_id == prompt_id)
            .options(joinedload(Evaluation.test_case))
            .order_by(Evaluation.timestamp.desc())
            .all()
        )

    def get_by_test_case(self, test_case_id: str) -> list[Evaluation]:
        """
        Get all evaluations for a specific test case.

        Args:
            test_case_id: Test case ID

        Returns:
            List of evaluations
        """
        return (
            self.session.query(Evaluation)
            .filter(Evaluation.test_case_id == test_case_id)
            .order_by(Evaluation.timestamp.desc())
            .all()
        )

    def get_failed_tests_for_prompt(self, prompt_id: str, threshold: float = 7.0) -> list[Evaluation]:
        """
        Get all failed evaluations for a prompt.

        Args:
            prompt_id: Prompt ID
            threshold: Score threshold below which a test is considered failed

        Returns:
            List of failed evaluations with test cases loaded
        """
        return (
            self.session.query(Evaluation)
            .filter(Evaluation.prompt_id == prompt_id, Evaluation.overall_score < threshold)
            .options(joinedload(Evaluation.test_case))
            .all()
        )

    def get_all_for_run(self, run_id: int) -> list[Evaluation]:
        """
        Get all evaluations for a run.

        Args:
            run_id: Optimization run ID

        Returns:
            All evaluations for the run
        """
        return self.session.query(Evaluation).filter(Evaluation.run_id == run_id).all()

    def count_for_run(self, run_id: int) -> int:
        """
        Count total evaluations for a run.

        Args:
            run_id: Optimization run ID

        Returns:
            Count of evaluations
        """
        return self.session.query(Evaluation).filter(Evaluation.run_id == run_id).count()
