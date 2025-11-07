"""Database session management and initialization."""

import logging
import os
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from prompt_optimizer.storage.models import Base

logger = logging.getLogger(__name__)


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """Enable foreign keys for SQLite."""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.close()


class Database:
    """Database connection and session management."""

    def __init__(self, db_path: str | Path = "prompt_optimizer/data/storage/optimizer.db"):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        # Create engine
        self.engine = create_engine(
            f"sqlite:///{self.db_path}",
            echo=False,  # Set to True for SQL query logging
            connect_args={"check_same_thread": False},  # Allow multi-threaded access
        )

        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine,
        )

        # Create tables
        self._init_db()

        logger.info(f"Database initialized at {self.db_path}")

    def _check_schema_compatible(self) -> bool:
        """
        Check if the existing database schema is compatible with current models.

        Returns:
            True if schema is compatible or DB doesn't exist, False if incompatible
        """
        if not self.db_path.exists():
            return True

        try:
            inspector = inspect(self.engine)

            # Check if optimization_runs table has the correct structure
            if "optimization_runs" in inspector.get_table_names():
                columns = {col["name"] for col in inspector.get_columns("optimization_runs")}
                # Check for required columns that should exist in new schema
                required_columns = {"id", "task_description", "started_at", "status"}
                # Check for columns that should NOT exist in new schema
                forbidden_columns = {"current_stage"}  # Removed in refactoring

                if not required_columns.issubset(columns):
                    logger.warning(
                        "Schema incompatible: missing required columns in optimization_runs"
                    )
                    return False

                if forbidden_columns.intersection(columns):
                    logger.warning("Schema incompatible: found old columns that should be removed")
                    return False

            # Check if prompts table exists and has run_id column
            if "prompts" in inspector.get_table_names():
                columns = {col["name"] for col in inspector.get_columns("prompts")}
                if "run_id" not in columns:
                    logger.warning("Schema incompatible: prompts table missing run_id column")
                    return False

            # Check that stage_results table does NOT exist (removed in refactoring)
            if "stage_results" in inspector.get_table_names():
                logger.warning(
                    "Schema incompatible: stage_results table exists but should be removed"
                )
                return False

            return True
        except Exception as e:
            logger.warning(f"Error checking schema compatibility: {e}")
            return False

    def _init_db(self) -> None:
        """Create all tables, recreating DB if schema is incompatible."""
        # Check if existing schema is compatible
        if not self._check_schema_compatible():
            logger.warning(f"Incompatible database schema detected at {self.db_path}")
            logger.warning("Dropping old database and creating new schema...")

            # Close any connections
            self.engine.dispose()

            # Delete old database file
            if self.db_path.exists():
                backup_path = self.db_path.with_suffix(".db.old")
                logger.info(f"Backing up old database to {backup_path}")
                os.rename(self.db_path, backup_path)

            # Recreate engine
            self.engine = create_engine(
                f"sqlite:///{self.db_path}",
                echo=False,
                connect_args={"check_same_thread": False},
            )
            self.SessionLocal = sessionmaker(
                autocommit=False,
                autoflush=False,
                bind=self.engine,
            )

        # Create all tables
        Base.metadata.create_all(bind=self.engine)
        self._create_indexes()

    def _create_indexes(self) -> None:
        """Create additional indexes for performance."""
        with self.engine.begin() as conn:
            # Indexes for common queries
            conn.execute(
                text("CREATE INDEX IF NOT EXISTS idx_prompts_run_stage ON prompts(run_id, stage)")
            )
            # Separate indexes for quick_score and rigorous_score for optimal query performance
            conn.execute(
                text("CREATE INDEX IF NOT EXISTS idx_prompts_quick_score ON prompts(quick_score DESC)")
            )
            conn.execute(
                text("CREATE INDEX IF NOT EXISTS idx_prompts_rigorous_score ON prompts(rigorous_score DESC)")
            )
            conn.execute(
                text("CREATE INDEX IF NOT EXISTS idx_prompts_track ON prompts(run_id, track_id)")
            )
            conn.execute(
                text("CREATE INDEX IF NOT EXISTS idx_evaluations_run ON evaluations(run_id)")
            )
            conn.execute(
                text("CREATE INDEX IF NOT EXISTS idx_evaluations_prompt ON evaluations(prompt_id)")
            )
            conn.execute(
                text("CREATE INDEX IF NOT EXISTS idx_evaluations_test ON evaluations(test_case_id)")
            )
            conn.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS idx_test_cases_run_stage ON test_cases(run_id, stage)"
                )
            )

    def get_session(self) -> Session:
        """
        Get a new database session.

        Returns:
            SQLAlchemy session instance
        """
        return self.SessionLocal()

    @contextmanager
    def session_scope(self) -> Generator[Session]:
        """
        Provide a transactional scope around a series of operations.

        Usage:
            with db.session_scope() as session:
                session.add(obj)
                # commit happens automatically
        """
        session = self.get_session()
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()

    def close(self) -> None:
        """Close database connection."""
        self.engine.dispose()
