"""Repository for Prompt data access."""

from typing import TYPE_CHECKING

from sqlalchemy.orm import Session, joinedload

from prompt_optimizer.storage.models import Prompt

if TYPE_CHECKING:
    from typing import List


class PromptRepository:
    """Data access layer for prompts."""

    def __init__(self, session: Session):
        """Initialize repository with database session."""
        self.session = session

    def save(self, prompt: Prompt) -> Prompt:
        """
        Save or update a prompt.

        Args:
            prompt: Prompt instance to save

        Returns:
            Saved prompt instance
        """
        self.session.merge(prompt)
        self.session.commit()
        return prompt

    def get_by_id(self, prompt_id: str) -> Prompt | None:
        """
        Get prompt by ID.

        Args:
            prompt_id: Prompt ID

        Returns:
            Prompt instance or None
        """
        return self.session.query(Prompt).filter(Prompt.id == prompt_id).first()

    def get_by_stage(self, run_id: int, stage: str) -> list[Prompt]:
        """
        Get all prompts for a specific stage in a run.

        Args:
            run_id: Optimization run ID
            stage: Stage name (e.g., 'initial', 'quick_filter', 'rigorous', 'refined')

        Returns:
            List of prompts ordered by score descending
        """
        return (
            self.session.query(Prompt)
            .filter(Prompt.run_id == run_id, Prompt.stage == stage)
            .order_by(Prompt.average_score.desc().nullslast())
            .all()
        )

    def get_top_k(self, run_id: int, stage: str, k: int) -> list[Prompt]:
        """
        Get top K prompts by score for a stage.

        Args:
            run_id: Optimization run ID
            stage: Stage name
            k: Number of top prompts to return

        Returns:
            List of top K prompts
        """
        return (
            self.session.query(Prompt)
            .filter(Prompt.run_id == run_id, Prompt.stage == stage)
            .order_by(Prompt.average_score.desc().nullslast())
            .limit(k)
            .all()
        )

    def get_by_track(self, run_id: int, track_id: int) -> list[Prompt]:
        """
        Get all prompts from a refinement track.

        Args:
            run_id: Optimization run ID
            track_id: Refinement track ID

        Returns:
            List of prompts ordered by iteration
        """
        return (
            self.session.query(Prompt)
            .filter(Prompt.run_id == run_id, Prompt.track_id == track_id)
            .order_by(Prompt.iteration)
            .all()
        )

    def get_original_prompt(self, run_id: int) -> Prompt | None:
        """
        Get the original system prompt for a run.

        Args:
            run_id: Optimization run ID

        Returns:
            Original prompt or None
        """
        return (
            self.session.query(Prompt)
            .filter(Prompt.run_id == run_id, Prompt.is_original_system_prompt.is_(True))
            .first()
        )

    def get_with_evaluations(self, prompt_id: str) -> Prompt | None:
        """
        Get prompt with all its evaluations eagerly loaded.

        Args:
            prompt_id: Prompt ID

        Returns:
            Prompt with evaluations or None
        """
        return (
            self.session.query(Prompt)
            .filter(Prompt.id == prompt_id)
            .options(joinedload(Prompt.evaluations))
            .first()
        )

    def get_all_for_run(self, run_id: int) -> list[Prompt]:
        """
        Get all prompts for a run.

        Args:
            run_id: Optimization run ID

        Returns:
            All prompts for the run
        """
        return self.session.query(Prompt).filter(Prompt.run_id == run_id).all()
