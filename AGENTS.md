# Project Agent Playbook

## Mission
- Safeguard the retrieval-assisted interview assistant: preserve existing workflows, ensure conversations stay logged, and keep responses grounded in user career data.
- Before coding, confirm intended behaviour, required inputs, and tests with the requester; surface blockers (missing config, sandbox limits) early.

## Architecture Map
- `src/app.py` → Entry point; initializes config, creates RAG engine via factory, and launches UI.
- `src/launchers/` → UI abstraction layer (Gradio or CLI) selected via `LAUNCHER_TYPE` env var.
  - `factory.py` → Creates launcher instances based on configuration.
  - `gradio_launcher.py` → Web interface implementation.
  - `cli_launcher.py` → Terminal interface implementation.
- `src/rag_engine/engine.py` → Orchestrates retrieval, prompt formatting, history, and logging.
- `src/rag_engine/prompt_logger.py` → Persists question/prompt/answer tuples to timestamped logs.
- `src/llm/factory.py` → Chooses an LLM backend (MLX Qwen/Llama or OpenAI) based on `Config.model_provider`.
- `src/llm/mlx_llm.py` / `src/llm/openai_llm.py` → Concrete implementations; both expect a LangChain retriever and must set `last_prompt` for logging.
- `src/document_loader/` → Document loading and vector store management.
  - `huggingface_loader.py` → Builds/loads Chroma vector store from `career_data/`.
  - `factory.py` → Creates document loader instances (currently only HuggingFace).

## Core Guidelines
- **Reuse the singleton**: obtain configuration via `get_config()`; avoid constructing fresh `Config` objects outside tests so settings (model, paths, tokens) stay in sync across components.
- **Keep provider enums aligned**: the provider keys `{"qwen", "llama", "openai"}` must match `config.yaml`, environment variables (via `MODEL_PROVIDER`), and `PROVIDER_CONFIG` in `src/llm/factory.py`. Default provider is `"qwen"` if not specified. Update all touchpoints plus docs/tests together when introducing a new provider.
- **Preserve prompt plumbing**: `RAGEngine.generate_response()` expects `LLMInterface.invoke()` / `.stream()` to return strings, respect streaming, and update `self.last_prompt`. Do not remove `PromptLogger` writes or manual history management.
- **Handle mixed retriever outputs**: Retrieval can yield LangChain `Document` objects or plain strings. Any context formatting logic must treat both (see `src/llm/openai_llm.py:_format_docs` and `src/llm/mlx_llm.py:_format_docs`).
- **Respect logging contract**: Use `logging.getLogger(__name__)` and never introduce `print()` in library code. Initialisation routines should log model names, devices, and recoverable failures.
- **Keep text encoding consistent**: project tooling assumes ASCII for most files; UTF-8 is used only where essential (prompt logs in `prompts/`, career data in `career_data/`).

## Coding Standards
- Format with `ruff format src/ tests/` and lint via `ruff check src/ tests/`; line length is 100, strings default to double quotes (`pyproject.toml`).
- Maintain type hints on all functions; prefer `collections.abc` types for annotations.
- Organize imports standard-lib → third-party → local; remove unused imports.
- Add comments sparingly—only when intent is non-obvious.
- For configuration additions, modify `src/config.py`, `config.yaml`, `.env.example`, and documentation together.

## Testing & Validation
- Baseline checks: `pytest -v`, `ruff check`, and targeted component tests for new logic. Add unit tests when behaviour changes or bugs are fixed.
- **Current test debt**: `tests/test_config.py` still asserts `model_provider in ["local", "openai"]` but actual valid providers are `{"qwen", "llama", "openai"}`. This test needs updating.
- For MLX-dependent code, consider guard tests or feature flags so CI without Apple Silicon can still pass (mock imports when necessary).
- Test coverage reports are generated in `htmlcov/` via `pytest --cov=src --cov-report=html`.

## Workflow Tips
- When adding LLM backends:
  1. Implement a class inheriting `LLMInterface` in `src/llm/`.
  2. Extend `PROVIDER_CONFIG` in `src/llm/factory.py` with backend type and system prompt hook.
  3. Register configuration defaults in `config.yaml` under `model.<provider>` section.
  4. Update `Config._load_model_settings` in `src/config.py` to load the new provider settings.
  5. Supply retrieval-aware prompts that set `self.last_prompt` for logging.
  6. Update tests, documentation, and `.env.example` to reflect the new provider.
- When adding launcher types:
  1. Implement a class inheriting `BaseLauncher` in `src/launchers/`.
  2. Register in `create_launcher` factory function.
  3. Update `LAUNCHER_TYPE` documentation in `.env.example`.
- When tweaking RAG heuristics, keep `_is_out_of_scope` behaviour configurable if thresholds/keywords expand.
- To introduce new persistence (e.g., analytics), mirror `PromptLogger` patterns: lazy directory creation, UTF-8 writes, error suppression via logging.
- For vector store changes, ensure CLI entry points work: run `python -m src.document_loader.huggingface_loader` or use the VSCode task "Rebuild Vector Database".

## Data & Secrets
- `.env` carries `MODEL_PROVIDER` (values: `qwen`, `llama`, `openai`), `LAUNCHER_TYPE` (values: `gradio`, `cli`), `USER_NAME`, and API tokens; never hardcode secrets.
- `.env`, `career_data/`, `prompts/`, `vector_db/`, and `models/` are gitignored—do not remove protections.
- Respect `vector_db/` and `models/` as generated/cached content; rebuild vector DB when data schema changes.
- Career documents in `career_data/*.md` contain private information—never commit or expose publicly.

## Common Pitfalls
- Attempting to instantiate MLX models on non-Apple Silicon: guard with `IS_APPLE_SILICON` and raise helpful errors (keep guidance to install `mlx`/`mlx-lm`).
- Forgetting to update `PromptLogger.log` call sites after signature changes results in empty prompt history files.
- Breaking streaming: `RAGEngine` yields tokens; ensure new LLM backends stream iterables of strings and close generators cleanly.
- Neglecting doc updates (`README.md`, `.claude` guides) when behaviour visibly changes—users rely on those for setup.

## Tooling Notes
- Run inside the project virtual environment (`uv venv` + `source .venv/bin/activate`); assume dependencies and CLI tasks execute from that shell.
- Start the app with `uv run python -m src.app`; the launcher (Gradio or CLI) is selected via `LAUNCHER_TYPE` in `.env` (`gradio` by default, set to `cli` for terminal mode).
- Package manager: `uv`; prefer `uv pip` for dependency installs.
- Type checking: `pyright` (configured via `pyrightconfig.json`) or `mypy` if needed.
- Coverage reports live in `htmlcov/`; regenerate with `pytest --cov=src --cov-report=html`.

Keep this playbook nearby; update it whenever the development workflow, supported providers, or critical constraints change.
