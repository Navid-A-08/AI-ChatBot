"""
CLI entry point: the conversational loop.

This is intentionally thin — it wires together the pieces (config,
logging, prompt loading, context assembly, the Claude client) but
doesn't contain logic itself. Each piece is independently testable;
this module is mostly glue plus the terminal I/O.
"""

import logging

from ai_chatbot.config import get_settings
from ai_chatbot.context import ConversationHistory
from ai_chatbot.llm.claude_client import ClaudeClient
from ai_chatbot.logging_setup import setup_logging
from ai_chatbot.prompts import load_prompt

# How many previous turns to include in each API call. Hardcoded here
# for now; will become configurable (and experimentally varied) once
# Phase 4/8 compare different windowing strategies.
MAX_HISTORY_TURNS = 6


def run_chat_loop() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)
    logger = logging.getLogger(__name__)

    system_prompt = load_prompt("system_prompt")
    client = ClaudeClient(api_key=settings.anthropic_api_key, model=settings.anthropic_model)
    history = ConversationHistory()

    print("AI Chatbot (Phase 1) — type 'exit' or 'quit' to leave.\n")

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

        messages = history.windowed_messages(MAX_HISTORY_TURNS, user_input)

        try:
            reply = client.send(messages, system_prompt)
        except Exception:
            logger.exception("Error calling Claude API")
            print("Assistant: (error contacting the model — see logs)\n")
            continue

        print(f"Assistant: {reply}\n")
        history.add_turn(user_input, reply)


if __name__ == "__main__":
    run_chat_loop()
