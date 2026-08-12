"""
Logging configuration.

Why this exists:
    print() statements can't be filtered by severity, can't be redirected
    to a file in production, and give no context (timestamp, module,
    level) about where a message came from. This matters a lot in later
    phases: you'll want to log things like "retrieved 4 chunks in 320ms"
    or "memory write skipped: below relevance threshold" without those
    lines being mixed in with actual chat output the user sees.

How it works:
    Python's built-in `logging` module, configured once at startup via
    `setup_logging()`. Every module then does:

        import logging
        logger = logging.getLogger(__name__)

    `__name__` makes log lines self-identify their source module
    (e.g. "ai_chatbot.llm.claude_client"), which is invaluable once the
    codebase has more than one file.

Alternative considered:
    A third-party structured logging lib (structlog, loguru) — nicer
    ergonomics and easier JSON output, but stdlib logging is enough for
    now and requires no new dependency. Revisit if/when logs need to be
    machine-parsed for the evaluation framework (Phase 5).
"""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """
    Configure root logging once, at application startup.

    Args:
        level: one of "DEBUG", "INFO", "WARNING", "ERROR".
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    # Third-party libraries (httpx, anthropic's HTTP client, etc.) are
    # chatty at INFO/DEBUG. Keep them quiet unless we're actively
    # debugging network calls, so our own logs aren't drowned out.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
