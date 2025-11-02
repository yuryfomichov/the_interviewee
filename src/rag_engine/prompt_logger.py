"""Utility for logging questions, prompts, and answers during a RAG session."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


class PromptLogger:
    """Persist prompt/response pairs to a timestamped log file per session."""

    def __init__(self, directory: Path | None = None) -> None:
        self._directory = directory or Path("prompts")
        local_time = datetime.now().astimezone()
        self.session_id = local_time.strftime("%Y%m%d_%H%M%S_%z")
        self.log_path = self._directory / f"prompt_log_{self.session_id}.txt"

        self._directory.mkdir(parents=True, exist_ok=True)
        try:
            self.log_path.write_text("", encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Unable to initialize prompt log file: {exc}")
        else:
            logger.info(f"Prompt log initialized at {self.log_path}")

    def log(self, question: str, prompt: str | None, answer: str) -> None:
        """Persist the most recent interaction to the log file."""
        prompt_text = prompt or ""
        timestamp = datetime.now().astimezone().isoformat()
        entry = (
            f"[{timestamp}]\n"
            f"QUESTION:\n{question}\n\n"
            f"PROMPT:\n{prompt_text}\n\n"
            f"ANSWER:\n{answer}\n" + "-" * 80 + "\n"
        )
        try:
            self.log_path.write_text(entry, encoding="utf-8")
        except Exception as exc:
            logger.warning(f"Failed to log prompt interaction: {exc}")
