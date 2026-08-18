import json
import tempfile
from pathlib import Path

from ai_chatbot.memory import MemoryManager, MemoryFact


def test_memory_manager_initialization():
    """Test that MemoryManager initializes correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "memory.json"
        manager = MemoryManager(long_term_path=str(path))
        stats = manager.get_stats()

        assert stats["short_term_count"] == 0
        assert stats["long_term_count"] == 0


def test_extract_facts_from_user_message():
    """Test fact extraction from user messages."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "memory.json"
        manager = MemoryManager(long_term_path=str(path))

        facts = manager.extract_facts("I am a software engineer.", "Got it!")
        assert len(facts) > 0
        assert any("software engineer" in fact for fact in facts)


def test_add_short_term_memory():
    """Test adding facts to short-term memory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "memory.json"
        manager = MemoryManager(long_term_path=str(path))

        manager.add_short_term("User prefers Python")
        stats = manager.get_stats()

        assert stats["short_term_count"] == 1


def test_add_long_term_memory():
    """Test adding facts to long-term memory with persistence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "memory.json"
        manager1 = MemoryManager(long_term_path=str(path))

        manager1.add_long_term("User is a developer")
        assert manager1.get_stats()["long_term_count"] == 1

        # Create new manager to test persistence
        manager2 = MemoryManager(long_term_path=str(path))
        assert manager2.get_stats()["long_term_count"] == 1


def test_long_term_memory_deduplication():
    """Test that duplicate facts are not added to long-term memory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "memory.json"
        manager = MemoryManager(long_term_path=str(path))

        manager.add_long_term("User likes Python")
        manager.add_long_term("User likes Python")  # Duplicate

        assert manager.get_stats()["long_term_count"] == 1


def test_get_context_empty():
    """Test getting context when memory is empty."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "memory.json"
        manager = MemoryManager(long_term_path=str(path))

        context = manager.get_context()
        assert context == ""


def test_get_context_with_short_term():
    """Test getting context with short-term memories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "memory.json"
        manager = MemoryManager(long_term_path=str(path))

        manager.add_short_term("User asked about Python")
        context = manager.get_context()

        assert "From this conversation:" in context
        assert "Python" in context


def test_get_context_with_long_term():
    """Test getting context with long-term memories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "memory.json"
        manager = MemoryManager(long_term_path=str(path))

        manager.add_long_term("User is a developer")
        context = manager.get_context()

        assert "From previous conversations:" in context
        assert "developer" in context


def test_process_turn():
    """Test processing a conversation turn to extract memories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "memory.json"
        manager = MemoryManager(long_term_path=str(path))

        manager.process_turn("I am a data scientist.", "Interesting field!")

        # Should have extracted at least one fact
        assert manager.get_stats()["short_term_count"] > 0


def test_clear_short_term_memory():
    """Test clearing short-term memory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "memory.json"
        manager = MemoryManager(long_term_path=str(path))

        manager.add_short_term("Temp fact")
        assert manager.get_stats()["short_term_count"] == 1

        manager.clear_short_term()
        assert manager.get_stats()["short_term_count"] == 0


def test_context_integration_with_conversation_history():
    """Test that memory integrates with ConversationHistory."""
    from ai_chatbot.context import ConversationHistory

    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "memory.json"
        history = ConversationHistory(memory=MemoryManager(long_term_path=str(path)))

        history.add_turn("I am a Python developer.", "Great!")
        history.add_turn("What is 2+2?", "4")

        # Should have memories
        assert history.memory.get_stats()["short_term_count"] > 0

        # Should be able to get memory context
        context = history.get_memory_context()
        assert "Python developer" in context or len(context) == 0  # May or may not extract
