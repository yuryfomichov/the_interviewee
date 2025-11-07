"""SQLAlchemy ORM models for prompt optimizer storage."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import ForeignKey, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all ORM models."""

    pass


class OptimizationRun(Base):
    """Track an entire optimization run."""

    __tablename__ = "optimization_runs"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    task_description: Mapped[str] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(default=datetime.now)
    completed_at: Mapped[datetime | None] = mapped_column(default=None)
    champion_prompt_id: Mapped[str | None] = mapped_column(ForeignKey("prompts.id"), default=None)
    total_tests_run: Mapped[int | None] = mapped_column(default=None)
    status: Mapped[str] = mapped_column(default="running")  # running, completed, failed
    total_time_seconds: Mapped[float | None] = mapped_column(default=None)

    # Relationships
    prompts: Mapped[list[Prompt]] = relationship(back_populates="run", foreign_keys="Prompt.run_id")
    test_cases: Mapped[list[TestCase]] = relationship(back_populates="run")
    evaluations: Mapped[list[Evaluation]] = relationship(back_populates="run")


class Prompt(Base):
    """A prompt candidate at any stage of the pipeline."""

    __tablename__ = "prompts"

    id: Mapped[str] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("optimization_runs.id"))
    prompt_text: Mapped[str] = mapped_column(Text)
    stage: Mapped[str]  # initial, quick_filter, rigorous, refined
    strategy: Mapped[str | None] = mapped_column(default=None)
    quick_score: Mapped[float | None] = mapped_column(default=None)
    rigorous_score: Mapped[float | None] = mapped_column(default=None)
    iteration: Mapped[int] = mapped_column(default=0)
    track_id: Mapped[int | None] = mapped_column(default=None)
    parent_prompt_id: Mapped[str | None] = mapped_column(
        ForeignKey("prompts.id"), default=None
    )  # Refinement lineage
    is_original_system_prompt: Mapped[bool] = mapped_column(default=False)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now)

    # Relationships
    run: Mapped[OptimizationRun] = relationship(back_populates="prompts", foreign_keys=[run_id])
    parent: Mapped[Prompt | None] = relationship(remote_side=[id], backref="children")
    evaluations: Mapped[list[Evaluation]] = relationship(back_populates="prompt")
    weaknesses: Mapped[list[WeaknessAnalysis]] = relationship(
        back_populates="prompt", order_by="WeaknessAnalysis.iteration"
    )


class TestCase(Base):
    """A test case for evaluating prompt performance."""

    __tablename__ = "test_cases"

    id: Mapped[str] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("optimization_runs.id"))
    input_message: Mapped[str] = mapped_column(Text)
    expected_behavior: Mapped[str] = mapped_column(Text)
    category: Mapped[str]  # core, edge, boundary, adversarial, consistency, format
    stage: Mapped[str]  # quick or rigorous
    created_at: Mapped[datetime] = mapped_column(default=datetime.now)

    # Relationships
    run: Mapped[OptimizationRun] = relationship(back_populates="test_cases")
    evaluations: Mapped[list[Evaluation]] = relationship(back_populates="test_case")


class Evaluation(Base):
    """Result of testing a prompt with a single test case."""

    __tablename__ = "evaluations"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("optimization_runs.id"))
    test_case_id: Mapped[str] = mapped_column(ForeignKey("test_cases.id"))
    prompt_id: Mapped[str] = mapped_column(ForeignKey("prompts.id"))
    model_response: Mapped[str] = mapped_column(Text)
    functionality: Mapped[int]
    safety: Mapped[int]
    consistency: Mapped[int]
    edge_case_handling: Mapped[int]
    reasoning: Mapped[str] = mapped_column(Text)
    overall_score: Mapped[float]
    timestamp: Mapped[datetime] = mapped_column(default=datetime.now)

    # Relationships
    run: Mapped[OptimizationRun] = relationship(back_populates="evaluations")
    test_case: Mapped[TestCase] = relationship(back_populates="evaluations")
    prompt: Mapped[Prompt] = relationship(back_populates="evaluations")


class WeaknessAnalysis(Base):
    """Weakness analysis from refinement iterations."""

    __tablename__ = "weakness_analyses"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    prompt_id: Mapped[str] = mapped_column(ForeignKey("prompts.id"))
    iteration: Mapped[int]
    description: Mapped[str] = mapped_column(Text)
    failed_test_ids: Mapped[str] = mapped_column(Text)  # JSON array as string
    failed_test_descriptions: Mapped[str] = mapped_column(Text)  # JSON array as string
    created_at: Mapped[datetime] = mapped_column(default=datetime.now)

    # Relationships
    prompt: Mapped[Prompt] = relationship(back_populates="weaknesses")
