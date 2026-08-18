import tempfile
from pathlib import Path

import pytest

from ai_chatbot.rag import RAGPipeline, RetrievedChunk


def test_retrieved_chunk_dataclass():
    """Test RetrievedChunk dataclass."""
    chunk = RetrievedChunk(
        content="Test content",
        source="test.txt",
        page=1,
        relevance_score=0.95,
    )

    assert chunk.content == "Test content"
    assert chunk.source == "test.txt"
    assert chunk.page == 1
    assert chunk.relevance_score == 0.95


def test_chunk_text_short():
    """Test chunking text that's shorter than chunk_size."""
    # Create a minimal pipeline just to test chunking
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline._chunk_size = 1000
    pipeline._chunk_overlap = 200

    text = "This is a short text."
    chunks = pipeline._chunk_text(text)

    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunk_text_long():
    """Test chunking text that's longer than chunk_size."""
    pipeline = RAGPipeline.__new__(RAGPipeline)
    pipeline._chunk_size = 100
    pipeline._chunk_overlap = 20

    text = "word " * 50  # 250 characters
    chunks = pipeline._chunk_text(text)

    assert len(chunks) > 1
    # All chunks should be non-empty
    assert all(chunk.strip() for chunk in chunks)


@pytest.mark.skip(reason="ChromaDB file locking issues on Windows - test manually")
def test_rag_pipeline_initialization():
    """Test that RAGPipeline initializes correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = RAGPipeline(
            documents_dir=f"{tmpdir}/docs",
            vector_store_dir=f"{tmpdir}/vector",
        )
        stats = pipeline.get_stats()

        assert stats["total_chunks"] == 0
        assert stats["chunk_size"] == 1000


@pytest.mark.skip(reason="ChromaDB file locking issues on Windows - test manually")
def test_ingest_text_file():
    """Test ingesting a text file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir) / "docs"
        docs_dir.mkdir()

        # Create a test file
        test_file = docs_dir / "test.txt"
        test_file.write_text("This is a test document with some content.")

        pipeline = RAGPipeline(
            documents_dir=str(docs_dir),
            vector_store_dir=f"{tmpdir}/vector",
        )

        chunks_ingested = pipeline.ingest_document(test_file)

        assert chunks_ingested == 1
        assert pipeline.get_stats()["total_chunks"] == 1


@pytest.mark.skip(reason="ChromaDB file locking issues on Windows - test manually")
def test_retrieve_empty():
    """Test retrieval when no documents are indexed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        pipeline = RAGPipeline(
            documents_dir=f"{tmpdir}/docs",
            vector_store_dir=f"{tmpdir}/vector",
        )

        results = pipeline.retrieve("test query")

        assert len(results) == 0


@pytest.mark.skip(reason="ChromaDB file locking issues on Windows - test manually")
def test_retrieve_with_documents():
    """Test retrieval after indexing documents."""
    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir) / "docs"
        docs_dir.mkdir()

        # Create and ingest a test file
        test_file = docs_dir / "test.txt"
        test_file.write_text("Python is a programming language used for data science and web development.")

        pipeline = RAGPipeline(
            documents_dir=str(docs_dir),
            vector_store_dir=f"{tmpdir}/vector",
        )
        pipeline.ingest_document(test_file)

        # Retrieve
        results = pipeline.retrieve("programming language", top_k=1)

        assert len(results) == 1
        assert isinstance(results[0], RetrievedChunk)
        assert "Python" in results[0].content


@pytest.mark.skip(reason="ChromaDB file locking issues on Windows - test manually")
def test_context_integration_with_rag():
    """Test that RAG integrates with ConversationHistory."""
    from ai_chatbot.context import ConversationHistory

    with tempfile.TemporaryDirectory() as tmpdir:
        docs_dir = Path(tmpdir) / "docs"
        docs_dir.mkdir()

        # Create test document
        test_file = docs_dir / "knowledge.txt"
        test_file.write_text("The company was founded in 2020 and has 100 employees.")

        history = ConversationHistory(
            rag=RAGPipeline(
                documents_dir=str(docs_dir),
                vector_store_dir=f"{tmpdir}/vector",
            )
        )

        # Ingest document
        history.rag.ingest_document(test_file)

        # Retrieve context
        chunks = history.retrieve_documents("company history")
        assert len(chunks) > 0

        # Format context
        context = history.format_retrieved_context(chunks)
        assert "Relevant Documents" in context
