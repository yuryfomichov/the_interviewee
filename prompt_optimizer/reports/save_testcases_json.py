"""Save test cases to JSON file."""

import json
from pathlib import Path

import aiofiles

from prompt_optimizer.schemas import OptimizationResult


async def save_testcases_json(result: OptimizationResult, output_dir: str) -> Path:
    """
    Save all test cases (quick and rigorous) to JSON file.

    Args:
        result: Optimization result containing test cases
        output_dir: Directory to save the JSON file

    Returns:
        Path to saved JSON file
    """
    testcases_file = Path(output_dir) / "testcases.json"
    testcases_file.parent.mkdir(parents=True, exist_ok=True)

    # Prepare test cases data
    testcases_data = {
        "quick_tests": [
            {
                "id": test.id,
                "input_message": test.input_message,
                "expected_behavior": test.expected_behavior,
                "category": test.category,
            }
            for test in result.quick_tests
        ],
        "rigorous_tests": [
            {
                "id": test.id,
                "input_message": test.input_message,
                "expected_behavior": test.expected_behavior,
                "category": test.category,
            }
            for test in result.rigorous_tests
        ],
        "summary": {
            "total_quick_tests": len(result.quick_tests),
            "total_rigorous_tests": len(result.rigorous_tests),
            "total_tests": len(result.quick_tests) + len(result.rigorous_tests),
        },
    }

    # Write to JSON file with pretty formatting
    async with aiofiles.open(testcases_file, "w") as f:
        await f.write(json.dumps(testcases_data, indent=2, ensure_ascii=False))

    print(f"Test cases saved to: {testcases_file}")
    return testcases_file
