# Prompt Optimizer Pipeline Tests

Comprehensive end-to-end tests for the prompt optimizer pipeline.

## Quick Start

### Using Make (Traditional)
```bash
make test              # Run all tests
make test-verbose      # Verbose output
make test-coverage     # With coverage report
```

### Using Just (Modern)
```bash
just test              # Run all tests
just test-verbose      # Verbose output
just test-coverage     # With coverage report
just test-file test_full_pipeline.py  # Run single file
just test-match "pipeline"            # Run tests matching pattern
```

### Using pytest directly
```bash
pytest tests/                    # Run all tests
pytest tests/ -v                 # Verbose
pytest tests/ -vv -s             # Very verbose with print statements
pytest tests/test_full_pipeline.py  # Single file
pytest tests/ -k "minimal"       # Only tests matching "minimal"
```

## Test Organization

```
tests/
├── conftest.py                    # Shared fixtures
├── test_full_pipeline.py          # End-to-end pipeline tests
├── test_prompt_progression.py     # Stage transitions & promotion
├── test_stage_behavior.py         # Individual stage logic
├── test_refinement.py             # Iterative improvement
├── test_edge_cases.py             # Edge cases & error handling
└── helpers/
    ├── dummy_connector.py         # Fake model connector
    ├── fake_agents.py             # Fake LLM agent responses
    └── assertions.py              # Custom assertions
```

## Testing Philosophy

These tests use **dummy/fake implementations** instead of mocks:

- **DummyConnector**: Fast, deterministic model responses (no API calls)
- **Fake Agents**: Realistic LLM agent outputs without OpenAI SDK calls
- **Real Database**: Uses actual SQLite (temp files), not in-memory

This approach provides:
- ✅ Fast execution (< 10 seconds for full suite)
- ✅ No external dependencies (no API keys needed)
- ✅ Deterministic results (seeded randomness)
- ✅ Realistic data flow (tests actual pipeline logic)

## Test Categories

### Full Pipeline (`test_full_pipeline.py`)
- Complete 10-stage execution
- Multiple configuration sizes (minimal, realistic)
- Parallel vs sequential execution
- Database state verification

### Prompt Progression (`test_prompt_progression.py`)
- Stage transitions (initial → quick_filter → rigorous → refined)
- Score-based selection
- Top-K/Top-M advancement
- Champion selection

### Stage Behavior (`test_stage_behavior.py`)
- Individual stage execution
- Input/output validation
- Count verification
- Score calculation

### Refinement (`test_refinement.py`)
- Iterative improvement
- Early stopping
- Convergence threshold
- Track isolation

### Edge Cases (`test_edge_cases.py`)
- Original prompt tracking
- Sync vs async modes
- Empty results handling
- Run isolation

## Configuration Fixtures

Tests use different configs for different purposes:

- **minimal_config**: 3 prompts, 2 quick tests, 3 rigorous tests (fast smoke tests)
- **realistic_config**: 15 prompts, 7 quick tests, 50 rigorous tests (production-like)
- **parallel_config**: Minimal config with parallel execution enabled
- **early_stopping_config**: Tuned for testing convergence behavior

## Custom Assertions

Helper assertions in `helpers/assertions.py`:

```python
assert_prompts_in_stage(session, run_id, "initial", expected_count=15)
assert_all_prompts_have_scores(session, run_id, "quick_filter", "quick_score")
assert_top_k_selected(session, run_id, k=5, "quick_filter", "quick_score")
assert_refinement_tracks_exist(session, run_id, num_tracks=3)
assert_champion_is_best(result)
```

## Coverage

Run with coverage report:
```bash
just test-coverage
# or
pytest tests/ --cov=prompt_optimizer --cov-report=html
# Open htmlcov/index.html in browser
```

## Debugging Tests

Run a single test with full output:
```bash
pytest tests/test_full_pipeline.py::test_minimal_pipeline_completes -vv -s
```

Use breakpoints:
```python
import pdb; pdb.set_trace()  # Add in test
pytest tests/test_full_pipeline.py -s  # Run with -s to enable pdb
```

## Dependencies

Tests require:
- pytest
- pytest-asyncio
- pytest-cov (for coverage)

These are included in the project's dev dependencies.
