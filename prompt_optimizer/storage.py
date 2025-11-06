"""SQLite storage for tracking prompts, tests, and evaluations."""

import logging
import sqlite3
from datetime import datetime
from pathlib import Path

from prompt_optimizer.types import EvaluationScore, PromptCandidate, TestCase, TestResult

logger = logging.getLogger(__name__)


class Storage:
    """SQLite-based storage for optimization tracking."""

    def __init__(self, db_path: str | Path = "prompt_optimizer/data/optimizer.db"):
        """Initialize storage and create tables if needed."""
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._create_tables()
        logger.info(f"Storage initialized at {self.db_path}")

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection."""
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _create_tables(self) -> None:
        """Create database schema."""
        with self._get_connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS prompts (
                    id TEXT PRIMARY KEY,
                    prompt_text TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    strategy TEXT,
                    average_score REAL,
                    iteration INTEGER DEFAULT 0,
                    track_id INTEGER,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS test_cases (
                    id TEXT PRIMARY KEY,
                    input_message TEXT NOT NULL,
                    expected_behavior TEXT NOT NULL,
                    category TEXT NOT NULL,
                    stage TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS evaluations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    test_case_id TEXT NOT NULL,
                    prompt_id TEXT NOT NULL,
                    model_response TEXT NOT NULL,
                    functionality INTEGER NOT NULL,
                    safety INTEGER NOT NULL,
                    consistency INTEGER NOT NULL,
                    edge_case_handling INTEGER NOT NULL,
                    reasoning TEXT NOT NULL,
                    overall_score REAL NOT NULL,
                    timestamp TEXT NOT NULL,
                    FOREIGN KEY (test_case_id) REFERENCES test_cases(id),
                    FOREIGN KEY (prompt_id) REFERENCES prompts(id)
                );

                CREATE TABLE IF NOT EXISTS optimization_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_description TEXT NOT NULL,
                    started_at TEXT NOT NULL,
                    completed_at TEXT,
                    champion_prompt_id TEXT,
                    total_tests_run INTEGER,
                    FOREIGN KEY (champion_prompt_id) REFERENCES prompts(id)
                );

                CREATE INDEX IF NOT EXISTS idx_prompts_stage ON prompts(stage);
                CREATE INDEX IF NOT EXISTS idx_prompts_score ON prompts(average_score DESC);
                CREATE INDEX IF NOT EXISTS idx_evaluations_prompt ON evaluations(prompt_id);
                CREATE INDEX IF NOT EXISTS idx_evaluations_test ON evaluations(test_case_id);
            """)

    def save_prompt(self, prompt: PromptCandidate) -> None:
        """Save a prompt candidate."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO prompts
                (id, prompt_text, stage, strategy, average_score, iteration, track_id, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    prompt.id,
                    prompt.prompt_text,
                    prompt.stage,
                    prompt.strategy,
                    prompt.average_score,
                    prompt.iteration,
                    prompt.track_id,
                    prompt.created_at.isoformat(),
                ),
            )

    def save_test_case(self, test: TestCase, stage: str) -> None:
        """Save a test case."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO test_cases
                (id, input_message, expected_behavior, category, stage)
                VALUES (?, ?, ?, ?, ?)
                """,
                (test.id, test.input_message, test.expected_behavior, test.category, stage),
            )

    def save_evaluation(self, result: TestResult) -> None:
        """Save a test evaluation result."""
        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT INTO evaluations
                (test_case_id, prompt_id, model_response, functionality, safety,
                 consistency, edge_case_handling, reasoning, overall_score, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    result.test_case_id,
                    result.prompt_id,
                    result.model_response,
                    result.evaluation.functionality,
                    result.evaluation.safety,
                    result.evaluation.consistency,
                    result.evaluation.edge_case_handling,
                    result.evaluation.reasoning,
                    result.evaluation.overall,
                    result.timestamp.isoformat(),
                ),
            )

    def get_prompt_evaluations(self, prompt_id: str) -> list[TestResult]:
        """Get all evaluations for a specific prompt."""
        with self._get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM evaluations WHERE prompt_id = ?
                ORDER BY timestamp DESC
                """,
                (prompt_id,),
            ).fetchall()

        return [
            TestResult(
                test_case_id=row["test_case_id"],
                prompt_id=row["prompt_id"],
                model_response=row["model_response"],
                evaluation=EvaluationScore(
                    functionality=row["functionality"],
                    safety=row["safety"],
                    consistency=row["consistency"],
                    edge_case_handling=row["edge_case_handling"],
                    reasoning=row["reasoning"],
                    overall=row["overall_score"],
                ),
                timestamp=datetime.fromisoformat(row["timestamp"]),
            )
            for row in rows
        ]

    def start_optimization_run(self, task_description: str) -> int:
        """Start tracking an optimization run."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO optimization_runs (task_description, started_at)
                VALUES (?, ?)
                """,
                (task_description, datetime.now().isoformat()),
            )
            return cursor.lastrowid

    def complete_optimization_run(
        self, run_id: int, champion_prompt_id: str, total_tests: int
    ) -> None:
        """Mark an optimization run as complete."""
        with self._get_connection() as conn:
            conn.execute(
                """
                UPDATE optimization_runs
                SET completed_at = ?, champion_prompt_id = ?, total_tests_run = ?
                WHERE id = ?
                """,
                (datetime.now().isoformat(), champion_prompt_id, total_tests, run_id),
            )
