"""Generate prompts stage: Create diverse initial prompt variations."""

import uuid

from agents import Runner

from prompt_optimizer.agents.prompt_generator_agent import create_generator_agent
from prompt_optimizer.optimizer.base_stage import BaseStage
from prompt_optimizer.optimizer.context import RunContext
from prompt_optimizer.types import PromptCandidate


class GeneratePromptsStage(BaseStage):
    """Stage 1: Generate diverse initial prompt variations."""

    @property
    def name(self) -> str:
        """Return the stage name."""
        return "Generate Prompts"

    async def _run_async(self, context: RunContext) -> RunContext:
        """
        Generate initial prompt variations (async mode).

        Args:
            context: Run context with task_spec

        Returns:
            Updated context with initial_prompts populated
        """
        prompts = []

        # Include original system prompt if provided
        if context.task_spec.current_prompt:
            self._print_progress("Including original system prompt for testing...")
            original_prompt = PromptCandidate(
                id=f"original_system_prompt_{uuid.uuid4().hex[:8]}",
                prompt_text=context.task_spec.current_prompt,
                stage="initial",
                strategy="original_system_prompt",
                is_original_system_prompt=True,
            )
            prompts.append(original_prompt)
            self._print_progress("Added original system prompt to testing pipeline")

        # Generate additional variations
        num_to_generate = (
            self.config.num_initial_prompts - 1
            if context.task_spec.current_prompt
            else self.config.num_initial_prompts
        )

        self._print_progress(f"Generating {num_to_generate} diverse prompt variations...")

        generator = create_generator_agent(
            self.config.generator_llm, context.task_spec, num_to_generate
        )
        result = await Runner.run(generator, "Generate diverse system prompts")
        generated_prompts = self._parse_generated_prompts(result.final_output)
        prompts.extend(generated_prompts)

        self._print_progress(
            f"Generated {len(prompts)} total prompts "
            f"({1 if context.task_spec.current_prompt else 0} original + "
            f"{len(generated_prompts)} variations)"
        )

        context.initial_prompts = prompts
        return context

    async def _run_sync(self, context: RunContext) -> RunContext:
        """
        Generate initial prompt variations (sync mode - same as async for this stage).

        Args:
            context: Run context with task_spec

        Returns:
            Updated context with initial_prompts populated
        """
        # This stage doesn't benefit from parallel execution since it's a single agent call
        return await self._run_async(context)

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
