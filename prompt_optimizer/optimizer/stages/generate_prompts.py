"""Generate prompts stage: Create diverse initial prompt variations."""

from agents import Runner

from prompt_optimizer.agents.generator_agent import create_generator_agent
from prompt_optimizer.optimizer.base_stage import BaseStage
from prompt_optimizer.optimizer.context import RunContext
from prompt_optimizer.types import PromptCandidate


class GeneratePromptsStage(BaseStage):
    """Stage 1: Generate diverse initial prompt variations."""

    @property
    def name(self) -> str:
        """Return the stage name."""
        return "Generate Prompts"

    async def run(self, context: RunContext) -> RunContext:
        """
        Generate initial prompt variations.

        Args:
            context: Run context with task_spec

        Returns:
            Updated context with initial_prompts populated
        """
        self._print_progress(f"Generating {self.config.num_initial_prompts} diverse prompts...")

        generator = create_generator_agent(
            self.config.generator_llm, context.task_spec, self.config.num_initial_prompts
        )
        result = await Runner.run(generator, "Generate diverse system prompts")
        prompts = self._parse_generated_prompts(result.final_output)

        self._print_progress(f"Generated {len(prompts)} prompt variations")

        context.initial_prompts = prompts
        return context

    def _parse_generated_prompts(self, agent_output) -> list[PromptCandidate]:
        """Parse agent output into PromptCandidate objects."""
        prompts = []
        for item in agent_output.prompts:
            prompts.append(
                PromptCandidate(
                    id=item.id,
                    prompt_text=item.prompt_text,
                    stage="initial",
                    strategy=item.strategy,
                )
            )
        return prompts
