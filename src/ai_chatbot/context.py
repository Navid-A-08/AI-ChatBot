"""
Context assembly: turns raw conversation history into what actually
gets sent to the LLM.

Why this exists:
    This is the first real piece of "context engineering" in the
    pipeline (per the project's architecture doc: Context Selection
    happens before the LLM call). Even at this early phase, we do NOT
    just dump the entire conversation history into every API call.

How it works — sliding window + memory + RAG:
    Keep only the most recent N turns (a "turn" = one user message +
    one assistant reply). Older turns are dropped entirely, but
    important facts are extracted to memory (Phase 2).

    Memory context (short-term and long-term) is injected into the
    system prompt, so the model has access to relevant facts even
    when they've fallen out of the sliding window.

    When documents are available, relevant chunks are retrieved via
    RAG (Phase 3) and added to the context.

Alternatives (deferred, not implemented here):
    - Token-based windowing (vs turn-count-based): windowing by token
      budget is more precise than turn count, since turns vary wildly
      in length. Turn-count is used here for simplicity; token-based
      windowing is a natural Phase 4 (context compression) upgrade.
"""

from dataclasses import dataclass, field
from ai_chatbot.memory import MemoryManager
from ai_chatbot.rag import RAGPipeline, RetrievedChunk


@dataclass
class Turn:
    """One exchange: a user message and the assistant's reply."""

    user: str
    assistant: str


@dataclass
class ConversationHistory:
    """
    Holds the full conversation, and knows how to produce a windowed
    view of it for sending to the LLM, including memory context
    and retrieved document chunks.
    """

    turns: list[Turn] = field(default_factory=list)
    memory: MemoryManager = field(default_factory=MemoryManager)
    rag: RAGPipeline = field(default_factory=RAGPipeline)

    def add_turn(self, user_message: str, assistant_reply: str) -> None:
        self.turns.append(Turn(user=user_message, assistant=assistant_reply))
        # Process the turn to extract and store memories
        self.memory.process_turn(user_message, assistant_reply)

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

    def get_memory_context(self) -> str:
        """
        Get memory context to inject into the system prompt.

        Returns:
            Formatted string with relevant memories from short-term
            and long-term storage.
        """
        return self.memory.get_context()

    def retrieve_documents(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """
        Retrieve relevant document chunks for a query.

        Args:
            query: The search query (typically the user's message).
            top_k: Number of chunks to retrieve.

        Returns:
            List of RetrievedChunk objects.
        """
        return self.rag.retrieve(query, top_k=top_k)

    def format_retrieved_context(self, chunks: list[RetrievedChunk]) -> str:
        """
        Format retrieved chunks into a context string.

        Args:
            chunks: List of RetrievedChunk objects.

        Returns:
            Formatted string suitable for injection into the prompt.
        """
        if not chunks:
            return ""

        parts = ["## Relevant Documents\n"]
        for i, chunk in enumerate(chunks, 1):
            source_info = f"(Source: {chunk.source}"
            if chunk.page is not None:
                source_info += f", Page {chunk.page}"
            source_info += ")"

            parts.append(f"### Excerpt {i} {source_info}")
            parts.append(chunk.content)
            parts.append("")

        return "\n".join(parts)
