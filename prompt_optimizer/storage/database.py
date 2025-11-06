"""Database session management and initialization."""

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event
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

    def _init_db(self) -> None:
        """Create all tables."""
        Base.metadata.create_all(bind=self.engine)
        self._create_indexes()

    def _create_indexes(self) -> None:
        """Create additional indexes for performance."""
        with self.engine.connect() as conn:
            # Indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prompts_run_stage ON prompts(run_id, stage)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prompts_score ON prompts(average_score DESC)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_prompts_track ON prompts(run_id, track_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_evaluations_run ON evaluations(run_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_evaluations_prompt ON evaluations(prompt_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_evaluations_test ON evaluations(test_case_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_test_cases_run_stage ON test_cases(run_id, stage)")
            conn.commit()

    def get_session(self) -> Session:
        """
        Get a new database session.

        Returns:
            SQLAlchemy session instance
        """
        return self.SessionLocal()

    @contextmanager
    def session_scope(self) -> Generator[Session, None, None]:
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
