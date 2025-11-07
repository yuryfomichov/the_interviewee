"""Repository for TestCase data access."""

from sqlalchemy.orm import Session

from prompt_optimizer.storage.models import TestCase


class TestCaseRepository:
    """Data access layer for test cases."""

    def __init__(self, session: Session):
        """Initialize repository with database session."""
        self.session = session

    def save(self, test_case: TestCase) -> TestCase:
        """
        Save or update a test case.

        Args:
            test_case: TestCase instance to save

        Returns:
            Saved test case instance
        """
        self.session.merge(test_case)
        self.session.commit()
        return test_case

    def save_many(self, test_cases: list[TestCase]) -> None:
        """
        Save multiple test cases in bulk.

        Args:
            test_cases: List of TestCase instances
        """
        for test_case in test_cases:
            self.session.merge(test_case)
        self.session.commit()

    def get_by_id(self, test_id: str) -> TestCase | None:
        """
        Get test case by ID.

        Args:
            test_id: Test case ID

        Returns:
            TestCase instance or None
        """
        return self.session.query(TestCase).filter(TestCase.id == test_id).first()

    def get_by_stage(self, run_id: int, stage: str) -> list[TestCase]:
        """
        Get all test cases for a specific stage.

        Args:
            run_id: Optimization run ID
            stage: Stage name ('quick' or 'rigorous')

        Returns:
            List of test cases
        """
        return (
            self.session.query(TestCase)
            .filter(TestCase.run_id == run_id, TestCase.stage == stage)
            .order_by(TestCase.created_at)
            .all()
        )

    def get_all_for_run(self, run_id: int) -> list[TestCase]:
        """
        Get all test cases for a run.

        Args:
            run_id: Optimization run ID

        Returns:
            All test cases for the run
        """
        return self.session.query(TestCase).filter(TestCase.run_id == run_id).all()

    def get_by_category(self, run_id: int, stage: str, category: str) -> list[TestCase]:
        """
        Get test cases by category.

        Args:
            run_id: Optimization run ID
            stage: Stage name
            category: Test category (e.g., 'core', 'edge', etc.)

        Returns:
            List of test cases matching the category
        """
        return (
            self.session.query(TestCase)
            .filter(
                TestCase.run_id == run_id,
                TestCase.stage == stage,
                TestCase.category == category,
            )
            .all()
        )
