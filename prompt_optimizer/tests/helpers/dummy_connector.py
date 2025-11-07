"""Dummy connector for testing - provides fast, deterministic responses."""

import hashlib
import random

from prompt_optimizer.connectors.base import BaseConnector


class DummyConnector(BaseConnector):
    """
    A dummy connector that generates fast, deterministic model responses.

    This connector doesn't make real API calls, making tests fast and reproducible.
    Responses are generated using simple heuristics based on the inputs.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the dummy connector.

        Args:
            seed: Random seed for deterministic responses
        """
        self.seed = seed
        self.call_count = 0

    async def test_prompt(self, system_prompt: str, message: str) -> str:
        """
        Generate a dummy response based on the system prompt and message.

        The response varies based on:
        - System prompt content (keywords, length)
        - User message content
        - Deterministic randomness (seeded)

        Args:
            system_prompt: The system prompt to use
            message: The user message to respond to

        Returns:
            A generated response string
        """
        self.call_count += 1

        # Create deterministic seed from inputs
        combined = f"{system_prompt}|{message}|{self.seed}|{self.call_count}"
        hash_value = int(hashlib.md5(combined.encode()).hexdigest()[:8], 16)
        rng = random.Random(hash_value)

        # Analyze system prompt for response characteristics
        has_detailed_instructions = len(system_prompt) > 200
        has_structure = any(
            marker in system_prompt.lower()
            for marker in ["rule", "format", "structure", "must", "should"]
        )
        has_examples = "example" in system_prompt.lower()

        # Generate response based on system prompt quality
        response_templates = []

        if has_detailed_instructions and has_structure:
            response_templates.extend(
                [
                    "Based on the guidelines provided, here's my analysis: {detail}",
                    "Following the specified format: {detail}",
                    "According to the rules outlined: {detail}",
                ]
            )
        elif has_detailed_instructions:
            response_templates.extend(
                [
                    "I understand the requirements. {detail}",
                    "Taking into account your instructions: {detail}",
                ]
            )
        else:
            response_templates.extend(
                [
                    "Here's my response: {detail}",
                    "{detail}",
                ]
            )

        # Add variety based on examples
        if has_examples:
            response_templates.append("As shown in the example: {detail}")

        # Generate detail content
        details = [
            "The solution involves careful consideration of the constraints.",
            "This approach balances efficiency with correctness.",
            "Multiple factors need to be evaluated here.",
            "The key insight is to prioritize the most important aspects.",
            "A systematic approach yields the best results.",
            "Context matters significantly in this scenario.",
            "The optimal strategy depends on the specific requirements.",
            "Quality assurance is essential for this task.",
        ]

        # Select template and detail
        template = rng.choice(response_templates)
        detail = rng.choice(details)
        response = template.format(detail=detail)

        # Add message-specific content
        if "?" in message:
            response += " This addresses your question directly."
        if any(word in message.lower() for word in ["how", "why", "what", "when"]):
            response += " Let me explain the reasoning behind this."

        # Vary response length based on prompt quality
        if has_detailed_instructions and has_structure:
            # Better prompts get more detailed responses
            additional_sentences = [
                " Furthermore, the approach ensures consistency.",
                " Additionally, this handles edge cases effectively.",
                " Moreover, the solution is scalable and maintainable.",
            ]
            response += rng.choice(additional_sentences)

        return response

    def reset_count(self) -> None:
        """Reset the call counter (useful between tests)."""
        self.call_count = 0
