.PHONY: test test-verbose test-coverage test-fast test-full test-single clean help

# Default target
help:
	@echo "Prompt Optimizer Test Commands"
	@echo "==============================="
	@echo ""
	@echo "make test          - Run all tests with minimal output"
	@echo "make test-verbose  - Run all tests with verbose output"
	@echo "make test-coverage - Run tests with coverage report"
	@echo "make test-fast     - Run only fast tests (minimal config)"
	@echo "make test-full     - Run all tests including realistic configs"
	@echo "make test-single   - Run a single test file (usage: make test-single FILE=test_full_pipeline.py)"
	@echo "make clean         - Clean test artifacts and cache"
	@echo ""

# Run all tests
test:
	pytest tests/ -v

# Run with verbose output
test-verbose:
	pytest tests/ -vv -s

# Run with coverage report
test-coverage:
	pytest tests/ --cov=prompt_optimizer --cov-report=term-missing --cov-report=html

# Run only fast tests (using minimal config fixtures)
test-fast:
	pytest tests/ -v -k "minimal"

# Run all tests including slow ones
test-full:
	pytest tests/ -v --durations=10

# Run a specific test file
test-single:
	@if [ -z "$(FILE)" ]; then \
		echo "Error: Please specify FILE=<test_file.py>"; \
		echo "Example: make test-single FILE=test_full_pipeline.py"; \
		exit 1; \
	fi
	pytest tests/$(FILE) -vv -s

# Clean test artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	rm -rf tests/__pycache__ 2>/dev/null || true
	@echo "Cleaned test artifacts"
