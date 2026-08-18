"""
RAG (Retrieval-Augmented Generation) pipeline.

Why this exists:
    The chatbot needs to answer questions about specific documents
    that weren't in its training data. RAG allows it to:
    1. Ingest documents (PDF, text, markdown)
    2. Split them into searchable chunks
    3. Embed chunks for semantic search
    4. Retrieve relevant chunks for each query
    5. Cite sources in responses

How it works:
    Documents are loaded from data/documents/, chunked into
    overlapping segments, embedded using ChromaDB's default
    embeddings, and stored in a vector database. When a user
    asks a question, we retrieve the top-K most relevant chunks
    and inject them into the context.

Alternatives considered:
    - Using an external vector DB (Pinecone, Weaviate): adds
      complexity and external dependencies. ChromaDB runs locally
      and is fine for a portfolio project.
    - Using sentence-transformers directly: ChromaDB handles
      embedding and storage together, simpler API.
    - Using LLM-based chunking: more intelligent but adds latency
      and cost. Simple character-based chunking works well enough.

Trade-offs accepted:
    - ChromaDB stores vectors locally, which means the vector store
      can grow large. For a portfolio project this is acceptable.
    - Simple chunking may break semantic units. Overlap helps but
      isn't perfect. This is a known limitation.
"""

import logging
from pathlib import Path
from dataclasses import dataclass

import chromadb
from pypdf import PdfReader

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    """A chunk of text retrieved from the vector store with metadata."""

    content: str
    source: str
    page: int | None = None
    relevance_score: float = 0.0


class RAGPipeline:
    """
    Manages document ingestion, chunking, embedding, and retrieval.
    """

    def __init__(
        self,
        documents_dir: str = "data/documents",
        vector_store_dir: str = "data/vector_store",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ) -> None:
        self._documents_dir = Path(documents_dir)
        self._vector_store_dir = Path(vector_store_dir)
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap

        # Initialize ChromaDB
        self._vector_store_dir.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=str(self._vector_store_dir))
        self._collection = self._client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )

    def ingest_document(self, file_path: Path) -> int:
        """
        Ingest a single document into the vector store.

        Args:
            file_path: Path to the document file.

        Returns:
            Number of chunks ingested.
        """
        logger.info("Ingesting document: %s", file_path)

        # Extract text based on file type
        if file_path.suffix.lower() == ".pdf":
            text = self._extract_pdf_text(file_path)
        elif file_path.suffix.lower() in {".txt", ".md", ".markdown"}:
            text = file_path.read_text(encoding="utf-8")
        else:
            logger.warning("Unsupported file type: %s", file_path.suffix)
            return 0

        # Chunk the text
        chunks = self._chunk_text(text)

        # Add to vector store
        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{file_path.name}_{i}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append({
                "source": file_path.name,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })

        if ids:
            self._collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info("Added %d chunks from %s", len(ids), file_path.name)

        return len(ids)

    def ingest_all_documents(self) -> int:
        """
        Ingest all documents from the documents directory.

        Returns:
            Total number of chunks ingested.
        """
        if not self._documents_dir.exists():
            logger.warning("Documents directory does not exist: %s", self._documents_dir)
            return 0

        total_chunks = 0
        supported_extensions = {".pdf", ".txt", ".md", ".markdown"}

        for file_path in self._documents_dir.rglob("*"):
            if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
                total_chunks += self.ingest_document(file_path)

        return total_chunks

    def retrieve(self, query: str, top_k: int = 3) -> list[RetrievedChunk]:
        """
        Retrieve the most relevant chunks for a query.

        Args:
            query: The search query.
            top_k: Number of chunks to return.

        Returns:
            List of RetrievedChunk objects sorted by relevance.
        """
        if self._collection.count() == 0:
            logger.info("No documents in vector store")
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(top_k, self._collection.count()),
        )

        chunks = []
        if results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 0

                chunk = RetrievedChunk(
                    content=doc,
                    source=metadata.get("source", "unknown"),
                    page=metadata.get("page"),
                    relevance_score=1 - distance,  # Convert distance to similarity
                )
                chunks.append(chunk)

        return chunks

    def _extract_pdf_text(self, file_path: Path) -> str:
        """Extract text from a PDF file."""
        try:
            reader = PdfReader(str(file_path))
            text_parts = []
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text:
                    text_parts.append(text)
            return "\n\n".join(text_parts)
        except Exception:
            logger.exception("Failed to extract text from PDF: %s", file_path)
            return ""

    def _chunk_text(self, text: str) -> list[str]:
        """
        Split text into overlapping chunks.

        Uses character-based chunking with overlap to maintain context
        across chunk boundaries.
        """
        if len(text) <= self._chunk_size:
            return [text] if text.strip() else []

        chunks = []
        start = 0

        while start < len(text):
            end = start + self._chunk_size

            # Try to break at a sentence or paragraph boundary
            if end < len(text):
                # Look for paragraph break
                para_break = text.rfind("\n\n", start, end)
                if para_break > start + self._chunk_size // 2:
                    end = para_break + 2
                else:
                    # Look for sentence break
                    for sep in [". ", ".\n", "! ", "? "]:
                        sent_break = text.rfind(sep, start, end)
                        if sent_break > start + self._chunk_size // 2:
                            end = sent_break + len(sep)
                            break

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)

            # Move start forward with overlap
            start = end - self._chunk_overlap if end < len(text) else end

        return chunks

    def get_stats(self) -> dict:
        """Return RAG pipeline statistics."""
        return {
            "documents_dir": str(self._documents_dir),
            "vector_store_dir": str(self._vector_store_dir),
            "total_chunks": self._collection.count(),
            "chunk_size": self._chunk_size,
            "chunk_overlap": self._chunk_overlap,
        }

    def clear(self) -> None:
        """Clear all documents from the vector store."""
        self._client.delete_collection("documents")
        self._collection = self._client.get_or_create_collection(
            name="documents",
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Cleared vector store")
