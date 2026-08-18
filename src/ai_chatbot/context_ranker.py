"""
Context ranking and compression strategies.

Why this exists:
    Phase 1-3 use simple turn-based windowing, which has limitations:
    - Turns vary wildly in length (token waste)
    - Older but relevant context is completely dropped
    - No way to prioritize important information

    This module provides intelligent context management:
    1. Token-based budgeting: Fit as much context as possible within
       a token budget, rather than counting turns.
    2. Importance scoring: Rank context chunks by relevance to the
       current query, so important older context survives.
    3. Summarization: Compress older turns into summaries to preserve
       information while reducing token usage.

How it works:
    ContextRanker takes the full conversation history and produces
    an optimized message list that fits within a token budget. It
    scores each piece of context (conversation chunks, memory facts,
    retrieved documents) by relevance to the current query, then
    greedily packs the highest-scoring items until the budget is full.

Alternatives considered:
    - Simple truncation: easiest but loses important context.
    - Sliding window (current Phase 1-3): better but still arbitrary.
    - LLM-based selection: most intelligent but adds latency and cost.
      The current approach uses simple heuristics (keyword overlap,
      recency) which is a good balance.

Trade-offs accepted:
    - Simple keyword-based relevance scoring may miss semantic similarity.
      Phase 8 (research comparison) will compare this with embedding-based
      scoring.
    - Summarization requires an extra LLM call, adding latency and cost.
      It's optional and only used when the context exceeds the budget.
"""

import re
from dataclasses import dataclass, field
from ai_chatbot.context import Turn


@dataclass
class ScoredChunk:
    """A piece of context with a relevance score."""

    content: str
    score: float
    chunk_type: str  # "turn", "memory", "document"
    metadata: dict = field(default_factory=dict)


class ContextRanker:
    """
    Intelligent context selection and compression.

    Ranks context pieces by relevance to the current query and
    packs them into a token budget.
    """

    def __init__(self, max_tokens: int = 8000) -> None:
        """
        Args:
            max_tokens: Maximum tokens for the assembled context.
                       Default 8000 is conservative for Claude.
        """
        self._max_tokens = max_tokens

    def rank_and_pack(
        self,
        turns: list[Turn],
        current_query: str,
        memory_context: str = "",
        document_chunks: list[str] = None,
    ) -> list[dict]:
        """
        Rank context by relevance and pack into token budget.

        Args:
            turns: Full conversation history.
            current_query: The current user message.
            memory_context: Memory facts to include.
            document_chunks: Retrieved document chunks to include.

        Returns:
            Optimized list of messages for the API.
        """
        if document_chunks is None:
            document_chunks = []

        # Score all context pieces
        scored_chunks: list[ScoredChunk] = []

        # Score conversation turns (pair user+assistant)
        for i in range(len(turns)):
            turn = turns[i]
            turn_text = f"User: {turn.user}\nAssistant: {turn.assistant}"
            score = self._score_relevance(turn_text, current_query)
            # Apply recency bonus (more recent = higher score)
            recency_bonus = i / max(len(turns), 1) * 0.2
            scored_chunks.append(ScoredChunk(
                content=turn_text,
                score=score + recency_bonus,
                chunk_type="turn",
                metadata={"turn_index": i},
            ))

        # Score memory chunks
        if memory_context:
            memory_chunks = self._split_into_chunks(memory_context, chunk_size=200)
            for chunk in memory_chunks:
                score = self._score_relevance(chunk, current_query)
                scored_chunks.append(ScoredChunk(
                    content=chunk,
                    score=score + 0.1,  # Small bonus for memory (persistent)
                    chunk_type="memory",
                ))

        # Score document chunks (highest priority - explicitly retrieved)
        for chunk in document_chunks:
            score = self._score_relevance(chunk, current_query)
            scored_chunks.append(ScoredChunk(
                content=chunk,
                score=score + 0.3,  # Higher bonus for retrieved documents
                chunk_type="document",
            ))

        # Sort by score (descending)
        scored_chunks.sort(key=lambda x: x.score, reverse=True)

        # Pack into token budget
        packed = self._pack_with_budget(scored_chunks, current_query)

        return packed

    def _score_relevance(self, text: str, query: str) -> float:
        """
        Score how relevant a text chunk is to the query.

        Uses simple keyword overlap. Phase 8 will compare this
        with embedding-based similarity.
        """
        if not query or not text:
            return 0.0

        # Extract keywords from query (simple: split on spaces, remove stopwords)
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "shall", "can", "need", "dare", "ought",
            "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after", "above", "below",
            "between", "out", "off", "over", "under", "again", "further", "then",
            "once", "here", "there", "when", "where", "why", "how", "all", "both",
            "each", "few", "more", "most", "other", "some", "such", "no", "nor",
            "not", "only", "own", "same", "so", "than", "too", "very", "just",
            "don", "now", "what", "which", "who", "whom", "this", "that", "these",
            "those", "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
            "you", "your", "yours", "yourself", "yourselves", "he", "him", "his",
            "himself", "she", "her", "hers", "herself", "it", "its", "itself",
            "they", "them", "their", "theirs", "themselves", "about", "up",
        }

        query_words = set(re.findall(r'\w+', query.lower())) - stopwords
        text_words = set(re.findall(r'\w+', text.lower()))

        if not query_words:
            return 0.0

        # Calculate Jaccard similarity
        intersection = query_words & text_words
        union = query_words | text_words

        return len(intersection) / len(union) if union else 0.0

    def _split_into_chunks(self, text: str, chunk_size: int = 200) -> list[str]:
        """Split text into smaller chunks for scoring."""
        if len(text) <= chunk_size:
            return [text] if text.strip() else []

        chunks = []
        sentences = re.split(r'(?<=[.!?])\s+', text)
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def _pack_with_budget(
        self, scored_chunks: list[ScoredChunk], current_query: str
    ) -> list[dict]:
        """
        Greedily pack scored chunks into the token budget.

        Always includes the current query at the end.
        """
        messages = []
        estimated_tokens = 0

        # Estimate tokens for the current query (always include)
        query_tokens = self._estimate_tokens(current_query)
        estimated_tokens += query_tokens

        # Add chunks until budget is full
        for chunk in scored_chunks:
            chunk_tokens = self._estimate_tokens(chunk.content)
            if estimated_tokens + chunk_tokens <= self._max_tokens:
                # Convert to message format
                if chunk.chunk_type == "turn":
                    # Parse the turn back into user/assistant messages
                    lines = chunk.content.split("\n", 1)
                    if len(lines) == 2:
                        user_msg = lines[0].replace("User: ", "")
                        assistant_msg = lines[1].replace("Assistant: ", "")
                        messages.append({"role": "user", "content": user_msg})
                        messages.append({"role": "assistant", "content": assistant_msg})
                elif chunk.chunk_type == "document":
                    # Wrap document chunks as user messages with context
                    messages.append({
                        "role": "user",
                        "content": f"[Retrieved document]\n{chunk.content}"
                    })
                elif chunk.chunk_type == "memory":
                    # Memory is handled in system prompt, but we can include
                    # important facts as context
                    messages.append({
                        "role": "user",
                        "content": f"[From memory]\n{chunk.content}"
                    })

                estimated_tokens += chunk_tokens

        # Always add the current query at the end
        messages.append({"role": "user", "content": current_query})

        return messages

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Uses simple word count heuristic. For production, use a
        proper tokenizer like tiktoken.
        """
        # Rough estimate: 1 token per 4 characters, or 1.3 tokens per word
        return max(1, len(text) // 4)

    def compress_turns(self, turns: list[Turn], target_tokens: int = 2000) -> str:
        """
        Compress older turns into a summary.

        This is a placeholder for LLM-based summarization.
        Currently just returns a simple concatenation.

        In a real implementation, this would call the LLM to
        generate a summary of the older turns.
        """
        if not turns:
            return ""

        # Simple compression: just list the topics discussed
        topics = []
        for turn in turns:
            # Extract first sentence as a topic indicator
            first_sentence = turn.user.split('.')[0]
            if len(first_sentence) > 50:
                first_sentence = first_sentence[:50] + "..."
            topics.append(first_sentence)

        return "Previous discussion topics: " + "; ".join(topics)
