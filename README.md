# Context-Aware AI Chatbot

> Status: **Phase 9 complete** (all features implemented).

A context-engineering-focused AI chatbot, built as a portfolio/research
project. Unlike a basic ChatGPT wrapper, the focus is on how context
(conversation history, memory, retrieved documents, tool output) is
selected, prioritized, compressed, and assembled before it reaches the LLM.

## Features

### Core Capabilities
- **Multi-turn conversations** with Claude API integration
- **Sliding window context** (Phase 1): Keeps recent conversation history
- **Memory system** (Phase 2): Short-term (session) and long-term (persistent) memory
- **RAG pipeline** (Phase 3): Document ingestion, chunking, embedding, and retrieval
- **Context ranking** (Phase 4): Intelligent context selection within token budgets
- **Evaluation framework** (Phase 5): Automated testing and metrics
- **Tool calling** (Phase 6): Calculator, time, text analysis tools
- **Multi-provider support** (Phase 7): Claude and OpenAI providers
- **Research framework** (Phase 8): Ablation studies and parameter sweeps

### Advanced Features
- **Automatic fact extraction**: Extracts and stores important information from conversations
- **Source citation**: RAG responses include document sources
- **Configurable context strategies**: Switch between windowed and ranked context
- **Experiment tracking**: Save and compare different configurations

## Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd ai-chatbot

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure API key
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY
```

### Running the Chatbot

```bash
# Basic usage
python -m ai_chatbot.cli

# With ranked context strategy
CONTEXT_STRATEGY=ranked python -m ai_chatbot.cli

# With OpenAI (if configured)
LLM_PROVIDER=openai OPENAI_API_KEY=your-key python -m ai_chatbot.cli
```

### Adding Documents for RAG

Place documents in the `data/documents/` directory:
- PDF files (`.pdf`)
- Text files (`.txt`)
- Markdown files (`.md`, `.markdown`)

Documents are automatically indexed on startup.

### Running Tests

```bash
# Run all tests
pytest -v

# Run specific test suites
pytest tests/test_memory.py -v
pytest tests/test_rag.py -v
pytest tests/test_context_ranker.py -v
```

### Running Evaluation

```python
from ai_chatbot.evaluation import load_default_dataset, Evaluator
from ai_chatbot.cli import create_chatbot

# Create chatbot
chatbot = create_chatbot()

# Load default test dataset
dataset = load_default_dataset()

# Evaluate
evaluator = Evaluator(chatbot)
report = evaluator.evaluate_dataset(dataset)

print(f"Average latency: {report.avg_latency_ms:.1f}ms")
print(f"Relevance score: {report.avg_relevance_score:.2f}")
print(f"Expected answers: {report.pct_contains_expected:.1f}%")
```

### Running Ablation Studies

```python
from ai_chatbot.research import ExperimentRunner, ExperimentConfig
from ai_chatbot.evaluation import load_default_dataset

# Create runner
runner = ExperimentRunner(your_chatbot_function)

# Load dataset
dataset = load_default_dataset()

# Run ablation study
results = runner.run_ablation_study(dataset)

# Compare results
comparison = runner.compare_results(results)
print(f"Best latency: {comparison['best_latency']}")
print(f"Best relevance: {comparison['best_relevance']}")
```

## Architecture

### Project Structure

```
ai-chatbot/
├── prompts/                    # Prompt templates
│   └── system_prompt.md
├── src/ai_chatbot/             # Main package
│   ├── config.py              # Configuration management
│   ├── context.py             # Context assembly
│   ├── context_ranker.py      # Intelligent context ranking
│   ├── memory.py              # Short-term and long-term memory
│   ├── rag.py                 # RAG pipeline
│   ├── evaluation.py          # Evaluation framework
│   ├── research.py            # Research comparison framework
│   ├── tools.py               # Tool definitions
│   ├── llm/                   # LLM providers
│   │   ├── base.py           # Provider abstraction
│   │   ├── claude_provider.py # Claude implementation
│   │   ├── openai_provider.py # OpenAI implementation
│   │   └── claude_client.py   # Legacy client wrapper
│   ├── cli.py                 # CLI entry point
│   ├── logging_setup.py       # Logging configuration
│   └── prompts.py             # Prompt loading
├── tests/                      # Test suite
├── data/
│   ├── documents/             # RAG document storage
│   ├── vector_store/          # ChromaDB vector index
│   ├── memory.json            # Long-term memory storage
│   └── experiments/           # Experiment results
├── requirements.txt
├── .env.example
└── README.md
```

### Context Engineering Pipeline

The system implements a complete context engineering pipeline:

1. **Context Selection** (context.py):
   - Sliding window for conversation history
   - Memory integration (short-term + long-term)
   - RAG retrieval for relevant documents

2. **Context Ranking** (context_ranker.py):
   - Relevance scoring based on query similarity
   - Recency weighting for conversation turns
   - Token budget optimization

3. **Context Assembly**:
   - System prompt with memory context
   - Ranked conversation history
   - Retrieved document chunks
   - Tool definitions (if enabled)

4. **Response Generation**:
   - Claude/OpenAI API call with assembled context
   - Tool use handling (if applicable)
   - Response post-processing

## Configuration

### Environment Variables

```bash
# Required
ANTHROPIC_API_KEY=your-anthropic-api-key

# Optional - Model selection
ANTHROPIC_MODEL=claude-sonnet-4-6
OPENAI_API_KEY=your-openai-key
OPENAI_MODEL=gpt-4
LLM_PROVIDER=claude  # or "openai"

# Optional - Context management
CONTEXT_STRATEGY=window  # or "ranked"
MAX_CONTEXT_TOKENS=8000
MAX_HISTORY_TURNS=6
RAG_TOP_K=3

# Optional - Logging
LOG_LEVEL=INFO
```

### Context Strategies

1. **Window Strategy** (default):
   - Keeps last N turns (configurable via MAX_HISTORY_TURNS)
   - Simple and predictable
   - Good for most use cases

2. **Ranked Strategy**:
   - Scores all context by relevance to current query
   - Packs within token budget
   - Better for complex queries requiring old context

## Research Applications

This project is designed for researching context engineering strategies:

### Ablation Studies

Measure the impact of each component:
- With/without memory
- With/without RAG
- Different context strategies
- Various window sizes

### Parameter Optimization

Sweep parameters to find optimal settings:
- Context token budgets
- RAG top-k values
- Memory retention periods

### Provider Comparison

Compare different LLM providers:
- Response quality
- Latency
- Cost
- Tool use capabilities

## Testing

### Test Coverage

- Unit tests for all core modules
- Integration tests for context pipeline
- Mocked API tests for LLM providers
- ChromaDB tests (skipped on Windows due to file locking)

### Running Specific Tests

```bash
# Memory system
pytest tests/test_memory.py -v

# RAG pipeline
pytest tests/test_rag.py -v

# Context ranking
pytest tests/test_context_ranker.py -v

# Evaluation framework
pytest tests/test_evaluation.py -v

# Tools
pytest tests/test_tools.py -v

# LLM providers
pytest tests/test_llm_providers.py -v

# Research framework
pytest tests/test_research.py -v
```

## License

Free to use. You may use, copy, modify, and share this project for any purpose.

## Contributing

Contributions are welcome! Please see CONTRIBUTING.md for guidelines.
