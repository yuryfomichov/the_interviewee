.PHONY: test test-verbose test-coverage test-fast test-full clean help

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
	@echo "make clean         - Clean test artifacts and cache"
	@echo ""
	@echo "NOTE: Run from the_interviewee directory (parent of prompt_optimizer)"
	@echo ""

# Run all tests
test:
	cd .. && python -m pytest prompt_optimizer/tests/ -v

# Run with verbose output
test-verbose:
	cd .. && python -m pytest prompt_optimizer/tests/ -vv -s

# Run with coverage report
test-coverage:
	cd .. && python -m pytest prompt_optimizer/tests/ --cov=prompt_optimizer --cov-report=term-missing --cov-report=html
	@echo ""
	@echo "Coverage report: htmlcov/index.html"

# Run only fast tests (using minimal config fixtures)
test-fast:
	cd .. && python -m pytest prompt_optimizer/tests/ -v -k "minimal"

# Run all tests including slow ones
test-full:
	cd .. && python -m pytest prompt_optimizer/tests/ -v --durations=10

# Clean test artifacts
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage" -delete 2>/dev/null || true
	@echo "Cleaned test artifacts"
