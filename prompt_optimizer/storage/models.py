"""SQLAlchemy ORM models for prompt optimizer storage."""

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy import Boolean, Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import declarative_base, relationship

if TYPE_CHECKING:
    from typing import List

Base = declarative_base()


class OptimizationRun(Base):
    """Track an entire optimization run."""

    __tablename__ = "optimization_runs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    task_description = Column(Text, nullable=False)
    started_at = Column(DateTime, nullable=False, default=datetime.now)
    completed_at = Column(DateTime, nullable=True)
    champion_prompt_id = Column(String, ForeignKey("prompts.id"), nullable=True)
    total_tests_run = Column(Integer, nullable=True)
    status = Column(String, default="running")  # running, completed, failed
    total_time_seconds = Column(Float, nullable=True)

    # Relationships
    prompts = relationship("Prompt", back_populates="run", foreign_keys="Prompt.run_id")
    test_cases = relationship("TestCase", back_populates="run")
    evaluations = relationship("Evaluation", back_populates="run")


class Prompt(Base):
    """A prompt candidate at any stage of the pipeline."""

    __tablename__ = "prompts"

    id = Column(String, primary_key=True)
    run_id = Column(Integer, ForeignKey("optimization_runs.id"), nullable=False)
    prompt_text = Column(Text, nullable=False)
    stage = Column(String, nullable=False)  # initial, quick_filter, rigorous, refined
    strategy = Column(String, nullable=True)
    average_score = Column(Float, nullable=True)
    quick_score = Column(Float, nullable=True)
    rigorous_score = Column(Float, nullable=True)
    iteration = Column(Integer, default=0)
    track_id = Column(Integer, nullable=True)
    parent_prompt_id = Column(String, ForeignKey("prompts.id"), nullable=True)  # Refinement lineage
    is_original_system_prompt = Column(Boolean, default=False)
    created_at = Column(DateTime, nullable=False, default=datetime.now)

    # Relationships
    run = relationship("OptimizationRun", back_populates="prompts", foreign_keys=[run_id])
    parent = relationship("Prompt", remote_side=[id], backref="children")
    evaluations = relationship("Evaluation", back_populates="prompt")
    weaknesses = relationship("WeaknessAnalysis", back_populates="prompt", order_by="WeaknessAnalysis.iteration")


class TestCase(Base):
    """A test case for evaluating prompt performance."""

    __tablename__ = "test_cases"

    id = Column(String, primary_key=True)
    run_id = Column(Integer, ForeignKey("optimization_runs.id"), nullable=False)
    input_message = Column(Text, nullable=False)
    expected_behavior = Column(Text, nullable=False)
    category = Column(String, nullable=False)  # core, edge, boundary, adversarial, consistency, format
    stage = Column(String, nullable=False)  # quick or rigorous
    created_at = Column(DateTime, default=datetime.now)

    # Relationships
    run = relationship("OptimizationRun", back_populates="test_cases")
    evaluations = relationship("Evaluation", back_populates="test_case")


class Evaluation(Base):
    """Result of testing a prompt with a single test case."""

    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    run_id = Column(Integer, ForeignKey("optimization_runs.id"), nullable=False)
    test_case_id = Column(String, ForeignKey("test_cases.id"), nullable=False)
    prompt_id = Column(String, ForeignKey("prompts.id"), nullable=False)
    model_response = Column(Text, nullable=False)
    functionality = Column(Integer, nullable=False)
    safety = Column(Integer, nullable=False)
    consistency = Column(Integer, nullable=False)
    edge_case_handling = Column(Integer, nullable=False)
    reasoning = Column(Text, nullable=False)
    overall_score = Column(Float, nullable=False)
    timestamp = Column(DateTime, nullable=False, default=datetime.now)

    # Relationships
    run = relationship("OptimizationRun", back_populates="evaluations")
    test_case = relationship("TestCase", back_populates="evaluations")
    prompt = relationship("Prompt", back_populates="evaluations")


class WeaknessAnalysis(Base):
    """Weakness analysis from refinement iterations."""

    __tablename__ = "weakness_analyses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    prompt_id = Column(String, ForeignKey("prompts.id"), nullable=False)
    iteration = Column(Integer, nullable=False)
    description = Column(Text, nullable=False)
    failed_test_ids = Column(Text, nullable=False)  # JSON array as string
    failed_test_descriptions = Column(Text, nullable=False)  # JSON array as string
    created_at = Column(DateTime, default=datetime.now)

    # Relationships
    prompt = relationship("Prompt", back_populates="weaknesses")
