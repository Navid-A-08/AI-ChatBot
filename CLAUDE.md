# AI Chatbot Project

## Project Overview

This is a context-engineering-focused AI chatbot built as a portfolio/research project. It demonstrates different strategies for managing conversation context, including memory systems, RAG, context ranking, and tool use.

## Key Architecture

The project follows a modular architecture with clear separation of concerns:

- **Context Pipeline**: context.py → context_ranker.py (assembles what goes to the LLM)
- **Memory System**: memory.py (short-term and long-term memory)
- **RAG Pipeline**: rag.py (document ingestion and retrieval)
- **LLM Providers**: llm/ directory (Claude, OpenAI abstractions)
- **Evaluation**: evaluation.py, research.py (testing and comparison frameworks)

## Development Guidelines

### Testing

- Run tests with: `pytest -v`
- All new features should include tests
- ChromaDB tests are skipped on Windows due to file locking issues

### Code Style

- Type hints are required for all functions
- Use dataclasses for structured data
- Keep functions focused and small
- Document complex algorithms with comments

### Adding New Features

1. Create the module in `src/ai_chatbot/`
2. Add tests in `tests/`
3. Update configuration in `config.py` if needed
4. Update this README with documentation

### LLM Provider Development

To add a new LLM provider:
1. Create `src/ai_chatbot/llm/new_provider.py`
2. Implement `BaseLLMProvider` interface
3. Register in `src/ai_chatbot/llm/__init__.py`
4. Add configuration options to `config.py`

## Common Commands

```bash
# Run the chatbot
python -m ai_chatbot.cli

# Run all tests
pytest -v

# Run specific test file
pytest tests/test_memory.py -v

# Install new dependencies
pip install new-package
# Then update requirements.txt
```

## Configuration

Key configuration options (set in .env):
- `ANTHROPIC_API_KEY`: Required for Claude
- `CONTEXT_STRATEGY`: "window" or "ranked"
- `LLM_PROVIDER`: "claude" or "openai"
- `MAX_HISTORY_TURNS`: Conversation history to keep (default: 6)
- `RAG_TOP_K`: Number of document chunks to retrieve (default: 3)

## Research Focus

This project is designed for studying context engineering:
- Ablation studies to measure component impact
- Parameter sweeps to optimize settings
- Provider comparison for quality/cost tradeoffs
