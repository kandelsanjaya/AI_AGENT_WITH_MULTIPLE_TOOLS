"""
rag.py
======
Retrieval-Augmented Generation (RAG) pipeline for EduSphere AI.

Responsibilities:
- Extract and chunk text from PDF documents.
- Build and query FAISS vector indexes via SentenceTransformer embeddings.
- Provide a clean public API used by the Streamlit UI.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import faiss
import numpy as np
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from .config import (
    EMBEDDING_MODEL_NAME,
    RAG_CHUNK_OVERLAP,
    RAG_CHUNK_SIZE,
    RAG_TOP_K,
)
from .exceptions import DocumentProcessingError, VectorStoreError

log = logging.getLogger("edusphere.rag")


# ---------------------------------------------------------------------------
# Singleton embedding model (cached by Streamlit)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="⚡ Loading Embedding Model…")
def load_embedding_model() -> SentenceTransformer:
    """Load and cache the SentenceTransformer model."""
    log.info("Loading embedding model: %s", EMBEDDING_MODEL_NAME)
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------

@dataclass
class VectorIndex:
    """Container for a FAISS index and its corresponding text chunks."""

    index: faiss.IndexFlatL2
    chunks: List[str] = field(default_factory=list)

    @property
    def size(self) -> int:
        return len(self.chunks)


# ---------------------------------------------------------------------------
# PDF Processing
# ---------------------------------------------------------------------------

def extract_pdf_chunks(
    file,
    chunk_size: int = RAG_CHUNK_SIZE,
    overlap: int = RAG_CHUNK_OVERLAP,
) -> List[str]:
    """
    Extract text from *file* (a file-like PDF object) and split it into
    overlapping chunks.

    Args:
        file: File-like object containing PDF data.
        chunk_size: Maximum characters per chunk.
        overlap: Character overlap between consecutive chunks.

    Returns:
        List of text chunks.

    Raises:
        DocumentProcessingError: If the PDF cannot be parsed.
    """
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size.")

    try:
        reader = PdfReader(file)
        full_text: List[str] = []
        for page in reader.pages:
            extracted = page.extract_text()
            if extracted:
                full_text.append(extracted)
        text = "\n".join(full_text)
    except Exception as exc:
        log.error("PDF extraction failed: %s", exc)
        raise DocumentProcessingError(f"Failed to read PDF: {exc}") from exc

    if not text.strip():
        raise DocumentProcessingError("The uploaded PDF appears to contain no extractable text.")

    chunks: List[str] = []
    start = 0
    while start < len(text):
        chunks.append(text[start : start + chunk_size])
        start += chunk_size - overlap

    log.info("Extracted %d chunks from PDF (%d chars).", len(chunks), len(text))
    return chunks


# ---------------------------------------------------------------------------
# Vector Store Manager
# ---------------------------------------------------------------------------

class VectorStoreManager:
    """Manages FAISS index creation and similarity search."""

    def __init__(self, model: Optional[SentenceTransformer] = None) -> None:
        self._model = model or load_embedding_model()

    # ------------------------------------------------------------------
    # Index building
    # ------------------------------------------------------------------

    def build_index(self, chunks: List[str]) -> VectorIndex:
        """
        Embed *chunks* and build a FAISS flat-L2 index.

        Args:
            chunks: List of text strings to embed.

        Returns:
            VectorIndex wrapping the FAISS index and chunks.

        Raises:
            VectorStoreError: If embedding or indexing fails.
        """
        if not chunks:
            raise VectorStoreError("Cannot build index from an empty chunk list.")

        try:
            embeddings = self._model.encode(
                chunks,
                show_progress_bar=False,
                batch_size=32,
            )
            vectors = np.array(embeddings, dtype="float32")
            dimension = vectors.shape[1]

            index = faiss.IndexFlatL2(dimension)
            index.add(vectors)
        except Exception as exc:
            log.error("FAISS index build failed: %s", exc)
            raise VectorStoreError(f"Failed to build vector index: {exc}") from exc

        log.info("Built FAISS index: %d vectors, dim=%d.", index.ntotal, dimension)
        return VectorIndex(index=index, chunks=chunks)

    # ------------------------------------------------------------------
    # Similarity search
    # ------------------------------------------------------------------

    def similarity_search(
        self,
        query: str,
        vi: VectorIndex,
        k: int = RAG_TOP_K,
    ) -> str:
        """
        Retrieve the top-*k* most similar chunks for *query*.

        Args:
            query: The user's search query.
            vi: A VectorIndex produced by :meth:`build_index`.
            k: Number of results to retrieve.

        Returns:
            A formatted string containing the retrieved context,
            or an empty string if the index is empty.

        Raises:
            VectorStoreError: If the FAISS search fails.
        """
        if vi.size == 0:
            return ""

        k = min(k, vi.size)

        try:
            q_vec = self._model.encode([query], show_progress_bar=False)
            q_arr = np.array(q_vec, dtype="float32")
            distances, indices = vi.index.search(q_arr, k)
        except Exception as exc:
            log.error("FAISS search failed: %s", exc)
            raise VectorStoreError(f"Similarity search failed: {exc}") from exc

        retrieved = [
            vi.chunks[i]
            for i in indices[0]
            if 0 <= i < vi.size
        ]

        if not retrieved:
            return ""

        separator = "\n\n" + "─" * 40 + "\n"
        context_block = separator.join(retrieved)
        return f"\n\n📄 **Relevant Document Context:**\n{context_block}\n"
