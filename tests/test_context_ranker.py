from ai_chatbot.context import Turn
from ai_chatbot.context_ranker import ContextRanker, ScoredChunk


def test_score_relevance_empty():
    """Test relevance scoring with empty inputs."""
    ranker = ContextRanker()
    score = ranker._score_relevance("", "test query")
    assert score == 0.0


def test_score_relevance_no_match():
    """Test relevance scoring with no keyword overlap."""
    ranker = ContextRanker()
    score = ranker._score_relevance("Python programming", "cooking recipes")
    assert score == 0.0


def test_score_relevance_partial_match():
    """Test relevance scoring with partial keyword overlap."""
    ranker = ContextRanker()
    score = ranker._score_relevance(
        "Python is a programming language",
        "Python programming"
    )
    assert 0.0 < score < 1.0


def test_score_relevance_exact_match():
    """Test relevance scoring with exact match."""
    ranker = ContextRanker()
    score = ranker._score_relevance("test", "test")
    assert score == 1.0


def test_split_into_chunks_short():
    """Test splitting short text."""
    ranker = ContextRanker()
    chunks = ranker._split_into_chunks("Short text.", chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0] == "Short text."


def test_split_into_chunks_long():
    """Test splitting long text."""
    ranker = ContextRanker()
    text = "First sentence. Second sentence. Third sentence. Fourth sentence."
    chunks = ranker._split_into_chunks(text, chunk_size=30)
    assert len(chunks) > 1


def test_estimate_tokens():
    """Test token estimation."""
    ranker = ContextRanker()
    tokens = ranker._estimate_tokens("Hello world, this is a test.")
    assert tokens > 0
    # Should be roughly 7-8 tokens for this sentence
    assert 5 <= tokens <= 15


def test_rank_and_pack_empty_history():
    """Test ranking with empty history."""
    ranker = ContextRanker()
    messages = ranker.rank_and_pack(
        turns=[],
        current_query="Hello",
    )

    assert len(messages) == 1
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "Hello"


def test_rank_and_pack_with_history():
    """Test ranking with conversation history."""
    ranker = ContextRanker(max_tokens=10000)
    turns = [
        Turn(user="What is Python?", assistant="Python is a programming language."),
        Turn(user="Tell me more.", assistant="It's used for web dev and data science."),
    ]

    messages = ranker.rank_and_pack(
        turns=turns,
        current_query="What are its main uses?",
    )

    # Should include history + current query
    assert len(messages) >= 3  # At least 2 from history + 1 current
    assert messages[-1]["content"] == "What are its main uses?"


def test_rank_and_pack_respects_token_budget():
    """Test that packing respects token budget."""
    ranker = ContextRanker(max_tokens=100)  # Very small budget
    turns = [
        Turn(user="Long question " * 20, assistant="Long answer " * 20),
        Turn(user="Another long question " * 20, assistant="Another long answer " * 20),
    ]

    messages = ranker.rank_and_pack(
        turns=turns,
        current_query="Short query",
    )

    # Current query should always be included
    assert messages[-1]["content"] == "Short query"


def test_scored_chunk_dataclass():
    """Test ScoredChunk dataclass."""
    chunk = ScoredChunk(
        content="Test content",
        score=0.85,
        chunk_type="turn",
        metadata={"turn_index": 2},
    )

    assert chunk.content == "Test content"
    assert chunk.score == 0.85
    assert chunk.chunk_type == "turn"
    assert chunk.metadata["turn_index"] == 2


def test_rank_and_pack_with_memory():
    """Test ranking with memory context."""
    ranker = ContextRanker(max_tokens=10000)
    turns = [Turn(user="Hi", assistant="Hello!")]

    messages = ranker.rank_and_pack(
        turns=turns,
        current_query="What do you remember about me?",
        memory_context="User is a Python developer",
    )

    # Should include the query
    assert messages[-1]["content"] == "What do you remember about me?"


def test_rank_and_pack_with_documents():
    """Test ranking with retrieved documents."""
    ranker = ContextRanker(max_tokens=10000)
    turns = []

    messages = ranker.rank_and_pack(
        turns=turns,
        current_query="Tell me about the project",
        document_chunks=[
            "The project was founded in 2020",
            "It has 100 employees",
        ],
    )

    # Should include documents and query
    assert len(messages) >= 3


def test_compress_turns_empty():
    """Test compressing empty turns."""
    ranker = ContextRanker()
    summary = ranker.compress_turns([])
    assert summary == ""


def test_compress_turns():
    """Test compressing turns into summary."""
    ranker = ContextRanker()
    turns = [
        Turn(user="What is Python?", assistant="A programming language."),
        Turn(user="How do I learn it?", assistant="Start with tutorials."),
    ]

    summary = ranker.compress_turns(turns)
    assert "Python" in summary
    assert "learn" in summary
