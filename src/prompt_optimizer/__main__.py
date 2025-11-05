"""Entry point for the AI Interviewee prompt optimizer.

Run the prompt optimization pipeline with:
    python -m src.prompt_optimizer

Requirements:
    - OpenAI API key set in .env file or OPENAI_API_KEY environment variable
    - prompt_optimizer dependencies installed
"""

import asyncio
import logging

from dotenv import load_dotenv

from .runner import run_optimization

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def main():
    """Main entry point."""
    asyncio.run(run_optimization())


if __name__ == "__main__":
    main()
