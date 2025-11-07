"""Base class for optimization stages."""

from abc import ABC, abstractmethod
from collections.abc import Callable

from prompt_optimizer.config import OptimizerConfig
from prompt_optimizer.connectors import BaseConnector
from prompt_optimizer.optimizer.context import RunContext
from prompt_optimizer.storage import Database


class BaseStage(ABC):
    """Base class for all optimization stages.

    All stages receive and return a RunContext object, ensuring
    a consistent interface across the pipeline.
    """

    def __init__(
        self,
        config: OptimizerConfig,
        database: Database,
        model_client: BaseConnector,
        progress_callback: Callable[[str], None] | None = None,
    ):
        """
        Initialize stage.

        Args:
            config: Optimizer configuration
            database: Database instance
            model_client: Connector for testing the target model
            progress_callback: Optional callback for progress messages
        """
        self.config = config
        self.database = database
        self.model_client = model_client
        self._print_progress = progress_callback or (lambda msg: None)

    async def run(self, context: RunContext) -> RunContext:
        """
        Execute the stage (dispatch to sync or async based on config).

        Args:
            context: Current run context with all pipeline state

        Returns:
            Updated run context with this stage's outputs
        """
        if self.config.parallel_execution:
            return await self._run_async(context)
        else:
            return await self._run_sync(context)

    @abstractmethod
    async def _run_async(self, context: RunContext) -> RunContext:
        """
        Execute the stage in parallel/async mode.

        Args:
            context: Current run context with all pipeline state

        Returns:
            Updated run context with this stage's outputs
        """
        pass

    @abstractmethod
    async def _run_sync(self, context: RunContext) -> RunContext:
        """
        Execute the stage in sequential/sync mode.

        Args:
            context: Current run context with all pipeline state

        Returns:
            Updated run context with this stage's outputs
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the stage name for logging."""
        pass
