"""
Memory system: short-term (session) and long-term (persistent) memory.

Why this exists:
    Phase 1's sliding window discards older turns completely. This is a
    fundamental limitation — users expect the chatbot to remember things
    from earlier in the conversation (short-term) and across sessions
    (long-term, like "I'm a software engineer" or "prefer concise answers").

    This module provides both:
    - Short-term memory: facts extracted from the current conversation,
      stored in-memory, available to context assembly.
    - Long-term memory: persistent facts stored to disk, loaded at
      session start, available across conversations.

How it works:
    Short-term memory uses simple keyword extraction and entity
    recognition to pull facts from each exchange (e.g., "I use Python"
    → user uses Python). This is intentionally simple for Phase 2 —
    more sophisticated extraction (LLM-based summarization) comes in
    Phase 4.

    Long-term memory persists to a JSON file, with automatic expiry
    and relevance scoring. Facts are scored by recency and frequency,
    then top-K are included in the system prompt.

Alternatives considered:
    - Vector-based memory (embedding + retrieval): more flexible but
      overlaps with Phase 3 (RAG). Deferred to avoid duplication.
    - LLM-based extraction: higher quality but adds latency and cost
      per turn. Reserved for Phase 4's context compression.
    - Full conversation storage: wasteful — most turns contain no
      memorable facts. Extract-and-store is more efficient.
"""

import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MemoryFact:
    """A single fact extracted from conversation or loaded from disk."""

    content: str
    source: str  # "short-term" or "long-term"
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0


class MemoryManager:
    """
    Manages short-term (session) and long-term (persistent) memory.

    Short-term memory lives only for the current session and is
    extracted from conversation turns. Long-term memory persists
    to disk and survives across sessions.
    """

    def __init__(self, long_term_path: str = "data/memory.json") -> None:
        self._short_term: list[MemoryFact] = []
        self._long_term_path = Path(long_term_path)
        self._long_term: list[MemoryFact] = []
        self._load_long_term()

    def _load_long_term(self) -> None:
        """Load long-term memory from disk if it exists."""
        if self._long_term_path.exists():
            try:
                data = json.loads(self._long_term_path.read_text(encoding="utf-8"))
                self._long_term = [MemoryFact(**item) for item in data]
                logger.info("Loaded %d long-term memories", len(self._long_term))
            except Exception:
                logger.exception("Failed to load long-term memory")

    def _save_long_term(self) -> None:
        """Persist long-term memory to disk."""
        try:
            self._long_term_path.parent.mkdir(parents=True, exist_ok=True)
            data = [
                {
                    "content": m.content,
                    "source": m.source,
                    "created_at": m.created_at,
                    "last_accessed": m.last_accessed,
                    "access_count": m.access_count,
                }
                for m in self._long_term
            ]
            self._long_term_path.write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
            logger.debug("Saved %d long-term memories", len(self._long_term))
        except Exception:
            logger.exception("Failed to save long-term memory")

    def extract_facts(self, user_message, assistant_reply) -> list[str]:
        """
        Extract memorable facts from a conversation turn.

        Uses simple heuristics for Phase 2. This will be upgraded to
        LLM-based extraction in Phase 4.

        Returns:
            List of extracted fact strings.
        """
        # Ensure inputs are strings
        user_message = str(user_message) if not isinstance(user_message, str) else user_message
        assistant_reply = str(assistant_reply) if not isinstance(assistant_reply, str) else assistant_reply

        facts = []

        # Simple patterns for fact extraction
        patterns = [
            ("I am ", ""),
            ("I'm ", ""),
            ("I use ", ""),
            ("I work ", ""),
            ("I prefer ", ""),
            ("I like ", ""),
            ("My name is ", ""),
            ("I need ", ""),
            ("I want ", ""),
            ("I'm a ", ""),
            ("I'm an ", ""),
        ]

        for prefix, _ in patterns:
            if prefix.lower() in user_message.lower():
                # Extract the sentence containing the pattern
                for sentence in user_message.split("."):
                    if prefix.lower() in sentence.lower():
                        fact = sentence.strip()
                        if fact and len(fact) > 5:
                            facts.append(fact)

        # Also capture if assistant suggests something user agrees with
        if any(
            word in assistant_reply.lower()
            for word in ["remember", "noted", "got it", "understood"]
        ):
            if len(user_message.split()) < 30:  # Short, memorable statements
                facts.append(user_message.strip())

        return facts

    def add_short_term(self, fact: str) -> None:
        """Add a fact to short-term (session) memory."""
        memory_fact = MemoryFact(content=fact, source="short-term")
        self._short_term.append(memory_fact)
        logger.debug("Added short-term memory: %s", fact[:50])

    def add_long_term(self, fact: str) -> None:
        """Add a fact to long-term (persistent) memory."""
        # Check for duplicates
        for existing in self._long_term:
            if existing.content.lower() == fact.lower():
                # Update access count instead of duplicating
                existing.access_count += 1
                existing.last_accessed = datetime.now().isoformat()
                return

        memory_fact = MemoryFact(content=fact, source="long-term")
        self._long_term.append(memory_fact)
        self._save_long_term()
        logger.debug("Added long-term memory: %s", fact[:50])

    def get_context(
        self, max_short_term: int = 10, max_long_term: int = 5
    ) -> str:
        """
        Generate a memory context string to inject into the system prompt.

        Args:
            max_short_term: max facts from current session
            max_long_term: max facts from persistent memory

        Returns:
            Formatted string suitable for injection into system prompt.
        """
        parts = []

        if self._short_term:
            recent_short = self._short_term[-max_short_term:]
            parts.append("From this conversation:")
            for fact in recent_short:
                parts.append(f"- {fact}")

        if self._long_term:
            # Score long-term memories by recency and frequency
            scored = []
            for fact in self._long_term:
                try:
                    created = datetime.fromisoformat(fact.created_at)
                    days_ago = (datetime.now() - created).days
                    score = fact.access_count + (1.0 / (days_ago + 1))
                    scored.append((score, fact))
                except Exception:
                    scored.append((0, fact))

            scored.sort(key=lambda x: x[0], reverse=True)
            top_long_term = scored[:max_long_term]

            if parts:
                parts.append("")
            parts.append("From previous conversations:")
            for _, fact in top_long_term:
                parts.append(f"- {fact.content}")

        return "\n".join(parts) if parts else ""

    def process_turn(self, user_message: str, assistant_reply: str) -> None:
        """
        Process a conversation turn to extract and store memories.

        Call this after each exchange to automatically populate memory.
        """
        facts = self.extract_facts(user_message, assistant_reply)

        for fact in facts:
            self.add_short_term(fact)

            # Promote to long-term if it seems important
            if self._should_promote_to_long_term(fact):
                self.add_long_term(fact)

    def _should_promote_to_long_term(self, fact: str) -> bool:
        """
        Heuristic to decide if a fact should be promoted to long-term memory.

        Phase 2 uses simple heuristics. Phase 4 will use LLM-based
        importance scoring.
        """
        # Promote facts about user identity, preferences, or work
        promotion_keywords = [
            "i am",
            "i'm a",
            "i'm an",
            "i work",
            "i use",
            "i prefer",
            "my name",
            "i need",
            "i want",
            "i like",
            "i don't like",
            "my job",
            "my role",
        ]

        fact_lower = fact.lower()
        return any(keyword in fact_lower for keyword in promotion_keywords)

    def clear_short_term(self) -> None:
        """Clear all short-term memory (e.g., at session end)."""
        self._short_term.clear()
        logger.info("Cleared short-term memory")

    def get_stats(self) -> dict:
        """Return memory statistics."""
        return {
            "short_term_count": len(self._short_term),
            "long_term_count": len(self._long_term),
            "long_term_path": str(self._long_term_path),
        }
