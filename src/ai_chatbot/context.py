"""
Context assembly: turns raw conversation history into what actually
gets sent to the LLM.

Why this exists:
    This is the first real piece of "context engineering" in the
    pipeline (per the project's architecture doc: Context Selection
    happens before the LLM call). Even at this early phase, we do NOT
    just dump the entire conversation history into every API call.

How it works — sliding window:
    Keep only the most recent N turns (a "turn" = one user message +
    one assistant reply). Older turns are dropped entirely.

    This is the simplest possible context management strategy, and
    that's intentional for Phase 1 — it establishes the pattern
    (there IS a context assembly step, it's not just "send everything")
    without yet solving the hard problem of *which* older context
    matters. That's exactly what later phases test:

    Alternatives (deferred, not implemented here):
    - Summarization: compress dropped turns into a running summary
      instead of discarding them outright. Preserves more information
      but costs an extra LLM call and introduces summarization drift.
    - Relevance-based retrieval: pull back specific older turns if
      they're relevant to the current message (this starts to overlap
      with memory/RAG — Phases 2-4).
    - Token-based windowing (vs turn-count-based): windowing by token
      budget is more precise than turn count, since turns vary wildly
      in length. Turn-count is used here for simplicity; token-based
      windowing is a natural Phase 4 (context compression) upgrade.

Trade-off being accepted right now:
    Dropped turns are gone completely — if the user references
    something from 10 messages ago and it fell out of the window, the
    model has no way to know. This is a real, known limitation of
    Phase 1, not an oversight. It's exactly the problem Phase 2
    (memory) and Phase 4 (better context strategies) exist to address.
"""

from dataclasses import dataclass, field


@dataclass
class Turn:
    """One exchange: a user message and the assistant's reply."""

    user: str
    assistant: str


@dataclass
class ConversationHistory:
    """
    Holds the full conversation, and knows how to produce a windowed
    view of it for sending to the LLM.
    """

    turns: list[Turn] = field(default_factory=list)

    def add_turn(self, user_message: str, assistant_reply: str) -> None:
        self.turns.append(Turn(user=user_message, assistant=assistant_reply))

    def windowed_messages(self, max_turns: int, current_user_message: str) -> list[dict]:
        """
        Build the message list to send to the API: the last `max_turns`
        turns, plus the new user message.

        Args:
            max_turns: how many previous turns to include. 0 means no
                       history at all — just the current message.
            current_user_message: the new message the user just sent
                                   (not yet added to `turns`, since we
                                   don't have the assistant's reply yet).

        Returns:
            A list of {"role": ..., "content": ...} dicts in the shape
            the Anthropic API expects.
        """
        recent_turns = self.turns[-max_turns:] if max_turns > 0 else []

        messages: list[dict] = []
        for turn in recent_turns:
            messages.append({"role": "user", "content": turn.user})
            messages.append({"role": "assistant", "content": turn.assistant})

        messages.append({"role": "user", "content": current_user_message})
        return messages
