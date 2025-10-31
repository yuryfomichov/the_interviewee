# AI Interviewee

An AI-powered interview practice assistant using RAG (Retrieval-Augmented Generation) to answer questions based on your career data. Built with Python, LangChain, MLX, and Gradio.

## Features

- **MLX & OpenAI Support**: Local models on Apple Silicon (MLX) or OpenAI API
- **RAG Architecture**: Retrieves relevant information from your career documents
- **Interactive Chat**: Clean Gradio web interface for interview practice
- **STAR Method Responses**: Generates answers using Situation, Task, Action, Result format
- **Source Citations**: Shows which documents were used to generate responses

## Quick Start

### 1. Setup Environment

```bash
# Install UV package manager (if not installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -r requirements.txt -r requirements-dev.txt
```

### 2. Add Your Career Data

Create markdown files in `career_data/`:

```bash
career_data/
├── cv.md                    # Your resume/CV
├── interview_answers.md     # Pre-written interview answers
└── career_stories.md        # Detailed project stories
```

**Example format** (use STAR method):

```markdown
# Professional Experience

## Senior Software Engineer at Company X (2020-2023)

### Key Projects

#### Microservices Migration
**Situation:** Legacy monolith causing deployment bottlenecks
**Task:** Lead migration to microservices architecture
**Action:** Designed service boundaries, implemented API gateway, trained team
**Result:** Reduced deployment time by 80%, improved scalability
```

### 3. Configure

Copy and edit the environment file:

```bash
cp .env.example .env
```

Edit `.env`:

```bash
# User Configuration
USER_NAME=YourName

# Model Configuration
MODEL_PROVIDER=local  # or 'openai'

# For MLX (Apple Silicon)
LOCAL_MODEL_NAME=mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit
DEVICE=mps

# For OpenAI (optional)
OPENAI_API_KEY=your_api_key_here

# Paths
CAREER_DATA_PATH=./career_data
VECTOR_DB_PATH=./vector_db
```

### 4. Build Vector Database

Index your career documents:

```bash
python src/document_loader.py --rebuild
```

### 5. Run Application

```bash
python src/app.py
```

Visit: http://localhost:7860

## Configuration

### Model Selection

#### Local Models (MLX - Apple Silicon Only)

Edit `config.yaml`:

```yaml
model:
  local:
    model_name: "mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit"
    device: "mps"
    max_new_tokens: 512
    temperature: 0.7
```

**Recommended models:**
- `mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit` - Excellent quality
- `mlx-community/Meta-Llama-3.1-8B-Instruct-4bit` - Fast, good quality

#### OpenAI API

Edit `config.yaml`:

```yaml
model:
  openai:
    model_name: "gpt-4o-mini"
    max_tokens: 1024
    temperature: 0.7
```

Set API key in `.env`:
```bash
OPENAI_API_KEY=your_key_here
```

### RAG Parameters

Fine-tune retrieval in `config.yaml`:

```yaml
rag:
  chunk_size: 800              # Size of document chunks
  chunk_overlap: 200           # Overlap between chunks
  top_k: 5                     # Number of chunks to retrieve
  relevance_threshold: 0.5     # Minimum similarity score
  embedding_model: "BAAI/bge-base-en-v1.5"
```

## Project Structure

```
interviewee/
├── src/
│   ├── config.py           # Configuration management
│   ├── document_loader.py  # Document loading and vector DB
│   ├── llm/                # LLM abstraction layer
│   │   ├── base.py         # Abstract LLM interface
│   │   ├── mlx_llm.py      # Apple Silicon MLX backend
│   │   ├── openai_llm.py   # OpenAI API backend
│   │   └── factory.py      # LLM factory function
│   ├── rag_engine.py       # RAG implementation
│   └── app.py              # Gradio web interface
├── tests/                  # Test files
├── career_data/            # Your career markdown files (gitignored)
├── vector_db/              # ChromaDB storage
└── config.yaml             # Configuration file
```

## Development

### VSCode Tasks

Available tasks (access via `Cmd/Ctrl+Shift+P` > "Tasks: Run Task"):

- **Run Gradio App** (`Ctrl+Shift+B`) - Start the web interface
- **Run Tests** - Execute pytest suite
- **Rebuild Vector Database** - Reindex career documents
- **Format Code (Ruff)** - Format Python code
- **Lint Code (Ruff)** - Check code quality

### Code Quality

```bash
# Format code
ruff format src/ tests/

# Lint code
ruff check src/

# Type check
mypy src/

# Run tests
pytest -v
```

## Troubleshooting

### "No markdown files found"
- Ensure `career_data/` directory exists and contains `.md` files
- Check `CAREER_DATA_PATH` in `.env` or `config.yaml`

### "Model not loading" or "Out of memory"
- Use a smaller quantized model (e.g., 4-bit instead of full precision)
- Try a smaller model size (e.g., 8B instead of 30B)
- Ensure you're on Apple Silicon for MLX support

### "MLX not available"
- MLX requires macOS and Apple Silicon (M1/M2/M3/M4)
- For other platforms, use OpenAI API instead

### Vector database issues
```bash
# Delete and rebuild
rm -rf vector_db/
python src/document_loader.py --rebuild
```

## System Requirements

### For Local Models (MLX)

- **Platform**: macOS with Apple Silicon (M1/M2/M3/M4)
- **RAM**: 16GB+ (32GB+ recommended for larger models)
- **Storage**: 10-30GB for model cache
- **macOS**: 12.3+ required for MLX

### For OpenAI API

- **Platform**: Any (macOS, Linux, Windows)
- **RAM**: 4GB+
- **Network**: Internet connection
- **API Key**: OpenAI account with credits

## Technology Stack

- **Framework**: LangChain for RAG pipeline
- **LLMs**: MLX (Apple Silicon), OpenAI API
- **Vector DB**: ChromaDB
- **Embeddings**: Sentence Transformers (BGE models)
- **UI**: Gradio
- **Tools**: UV (package manager), Ruff (linting/formatting)

## License

MIT License - feel free to use for personal interview practice!

## Acknowledgments

- Built with [LangChain](https://www.langchain.com/)
- MLX framework by [Apple ML Research](https://github.com/ml-explore/mlx)
- Models from [HuggingFace](https://huggingface.co/)
- UI powered by [Gradio](https://gradio.app/)
- Vector DB by [ChromaDB](https://www.trychroma.com/)

---

**Note**: This tool is for interview practice only. Keep your career data private and never commit the `career_data/` directory to public repositories.
