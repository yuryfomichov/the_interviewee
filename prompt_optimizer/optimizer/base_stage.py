"""Base class for optimization stages."""

from abc import ABC, abstractmethod
from collections.abc import Callable

from prompt_optimizer.config import OptimizerConfig
from prompt_optimizer.connectors import BaseConnector
from prompt_optimizer.optimizer.context import RunContext
from prompt_optimizer.storage import Storage


class BaseStage(ABC):
    """Base class for all optimization stages.

    All stages receive and return a RunContext object, ensuring
    a consistent interface across the pipeline.
    """

    def __init__(
        self,
        config: OptimizerConfig,
        storage: Storage,
        model_client: BaseConnector,
        progress_callback: Callable[[str], None] | None = None,
    ):
        """
        Initialize stage.

        Args:
            config: Optimizer configuration
            storage: Storage instance
            model_client: Connector for testing the target model
            progress_callback: Optional callback for progress messages
        """
        self.config = config
        self.storage = storage
        self.model_client = model_client
        self._print_progress = progress_callback or (lambda msg: None)

    @abstractmethod
    async def run(self, context: RunContext) -> RunContext:
        """
        Execute the stage.

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
