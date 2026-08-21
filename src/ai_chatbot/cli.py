"""
CLI entry point: the conversational loop.

This is intentionally thin — it wires together the pieces (config,
logging, prompt loading, context assembly, memory, RAG, the Claude
client) but doesn't contain logic itself. Each piece is independently
testable; this module is mostly glue plus the terminal I/O.

Phase 4 adds support for different context strategies:
- "window": Simple sliding window (Phase 1-3 behavior)
- "ranked": Intelligent ranking and packing within token budget
"""

import logging

from ai_chatbot.config import get_settings
from ai_chatbot.context import ConversationHistory
from ai_chatbot.context_ranker import ContextRanker
from ai_chatbot.llm.base import LLMConfig, default_provider_registry
from ai_chatbot.logging_setup import setup_logging
from ai_chatbot.prompts import load_prompt


def run_chat_loop() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    base_system_prompt = load_prompt("system_prompt")
    history = ConversationHistory()
    ranker = ContextRanker(max_tokens=settings.max_context_tokens)

    # Create LLM provider based on config
    if settings.llm_provider == "nvidia":
        llm_config = LLMConfig(
            model=settings.nvidia_model,
            api_key=settings.nvidia_api_key,
            max_tokens=1024,
            extra_params={"base_url": settings.nvidia_base_url},
        )
    elif settings.llm_provider == "openai":
        llm_config = LLMConfig(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            max_tokens=1024,
        )
    else:
        llm_config = LLMConfig(
            model=settings.anthropic_model,
            api_key=settings.anthropic_api_key,
            max_tokens=1024,
        )

    provider = default_provider_registry.create_provider(settings.llm_provider, llm_config)
    if provider is None:
        raise ValueError(f"Unknown LLM provider: {settings.llm_provider}")

    logger.info("Using LLM provider: %s (%s)", provider.provider_name, provider.model)

    # Ingest documents on startup if any exist
    logger.info("Ingesting documents...")
    chunks_ingested = history.rag.ingest_all_documents()
    if chunks_ingested > 0:
        logger.info("Ingested %d document chunks", chunks_ingested)
        print(f"Loaded {chunks_ingested} document chunks for RAG.\n")

    print(f"AI Chatbot — provider: {provider.provider_name}, model: {provider.model}")
    print(f"Context strategy: {settings.context_strategy}")
    print("Type 'exit' or 'quit' to leave.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        # Build system prompt with memory and RAG context
        prompt_parts = [base_system_prompt]

        # Add memory context
        memory_context = history.get_memory_context()
        if memory_context:
            prompt_parts.append("\n\n## Relevant Memory\n\n" + memory_context)

        # Add RAG context
        retrieved_chunks = history.retrieve_documents(user_input, top_k=settings.rag_top_k)
        rag_context = history.format_retrieved_context(retrieved_chunks)
        if rag_context:
            prompt_parts.append("\n\n" + rag_context)

        system_prompt = "".join(prompt_parts)

        # Assemble messages based on context strategy
        if settings.context_strategy == "ranked":
            # Use intelligent ranking and packing
            document_texts = [chunk.content for chunk in retrieved_chunks]
            messages = ranker.rank_and_pack(
                turns=history.turns,
                current_query=user_input,
                memory_context=memory_context,
                document_chunks=document_texts,
            )
        else:
            # Use simple sliding window (default)
            messages = history.windowed_messages(settings.max_history_turns, user_input)

        try:
            response = provider.send_message(messages, system_prompt)
            reply = response.text
        except Exception:
            logger.exception("Error calling LLM API")
            print("Assistant: (error contacting the model — see logs)\n")
            continue

        print(f"Assistant: {reply}\n")
        history.add_turn(user_input, reply)


if __name__ == "__main__":
    run_chat_loop()
