# Context-Aware AI Chatbot

> Status: **Phase 0 complete** (repo skeleton, config, logging).
> This README will grow into the full portfolio README as phases complete —
> for now it's a working setup doc.

## What this is

A context-engineering-focused AI chatbot, built as a portfolio/research
project. Unlike a basic ChatGPT wrapper, the focus is on how context
(conversation history, memory, retrieved documents, tool output) is
selected, prioritized, compressed, and assembled before it reaches the LLM.

Full architecture and phase plan: see `/docs` (added in a later phase).

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# then edit .env and set ANTHROPIC_API_KEY
```

## Running tests

```bash
pytest -v
```

## Project layout

```
ai-chatbot/
├── src/ai_chatbot/       # application package
│   ├── config.py         # typed settings loaded from env/.env
│   ├── logging_setup.py  # centralized logging config
│   └── llm/               # LLM client code (Phase 1)
├── tests/                 # pytest tests, mirrors src/ structure
├── data/
│   ├── documents/         # uploaded/ingested source docs (gitignored)
│   └── vector_store/      # embedded vector index (gitignored, regeneratable)
├── requirements.txt
└── .env.example
```

## Phase roadmap

| Phase | Status | Deliverable |
|---|---|---|
| 0 | ✅ Done | Repo skeleton, config, logging |
| 1 | ⏳ Next | Core conversational loop (Claude API + basic context assembly) |
| 2 | Planned | Memory system (short-term + long-term) |
| 3 | Planned | RAG pipeline (ingest → chunk → embed → retrieve → cite) |
| 4 | Planned | Context ranking + compression |
| 5 | Planned | Evaluation framework |
| 6 | Planned | Tool calling / basic agent |
| 7 | Planned | Multi-provider LLM abstraction |
| 8 | Planned | Research comparison (RAG/memory/tools ablation study) |
| 9 | Planned | Deployment + polish |
