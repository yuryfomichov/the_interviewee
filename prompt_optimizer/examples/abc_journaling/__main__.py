"""CLI entry point for the ABC journaling prompt optimizer example."""

from __future__ import annotations

import asyncio

from prompt_optimizer.examples.abc_journaling.optimize import main as run_optimizer


def main() -> None:
    """Execute the optimizer."""
    asyncio.run(run_optimizer())


if __name__ == "__main__":
    main()
