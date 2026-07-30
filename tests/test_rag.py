"""
tests/test_rag.py
=================
Unit tests for the EduSphere AI RAG pipeline.
"""

from __future__ import annotations

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.exceptions import DocumentProcessingError, VectorStoreError
from src.rag import VectorIndex, VectorStoreManager, extract_pdf_chunks


# ---------------------------------------------------------------------------
# extract_pdf_chunks — uses mocked PdfReader
# ---------------------------------------------------------------------------

class TestExtractPdfChunks:
    def _make_pdf_mock(self, text: str):
        """Create a mock PdfReader page with the given text."""
        page = MagicMock()
        page.extract_text.return_value = text
        reader = MagicMock()
        reader.pages = [page]
        return reader

    @patch("src.rag.PdfReader")
    def test_basic_extraction(self, mock_reader_cls):
        text = "A" * 2000
        mock_reader_cls.return_value = self._make_pdf_mock(text)

        chunks = extract_pdf_chunks(io.BytesIO(b"fake_pdf"))
        assert len(chunks) > 1
        assert all(isinstance(c, str) for c in chunks)

    @patch("src.rag.PdfReader")
    def test_chunk_size_respected(self, mock_reader_cls):
        text = "X" * 3000
        mock_reader_cls.return_value = self._make_pdf_mock(text)

        chunks = extract_pdf_chunks(io.BytesIO(b"fake"), chunk_size=500, overlap=50)
        for chunk in chunks[:-1]:  # last chunk may be smaller
            assert len(chunk) <= 500

    @patch("src.rag.PdfReader")
    def test_empty_pdf_raises(self, mock_reader_cls):
        page = MagicMock()
        page.extract_text.return_value = ""
        mock_reader_cls.return_value.pages = [page]

        with pytest.raises(DocumentProcessingError):
            extract_pdf_chunks(io.BytesIO(b"empty"))

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            extract_pdf_chunks(io.BytesIO(b"x"), chunk_size=100, overlap=200)

    @patch("src.rag.PdfReader", side_effect=Exception("corrupt"))
    def test_corrupt_pdf_raises(self, _mock):
        with pytest.raises(DocumentProcessingError, match="Failed to read PDF"):
            extract_pdf_chunks(io.BytesIO(b"corrupt"))


# ---------------------------------------------------------------------------
# VectorStoreManager — uses a mock embedding model
# ---------------------------------------------------------------------------

def _make_manager(dim: int = 8) -> VectorStoreManager:
    """Create a VectorStoreManager backed by a deterministic mock encoder."""
    mock_model = MagicMock()

    def fake_encode(texts, **kwargs):
        rng = np.random.default_rng(42)
        return rng.random((len(texts), dim)).astype("float32")

    mock_model.encode.side_effect = fake_encode
    return VectorStoreManager(model=mock_model)


class TestVectorStoreManager:
    def test_build_index_success(self):
        mgr = _make_manager()
        chunks = ["chunk one", "chunk two", "chunk three"]
        vi = mgr.build_index(chunks)

        assert isinstance(vi, VectorIndex)
        assert vi.size == 3
        assert vi.index.ntotal == 3

    def test_build_index_empty_raises(self):
        mgr = _make_manager()
        with pytest.raises(VectorStoreError):
            mgr.build_index([])

    def test_similarity_search_returns_string(self):
        mgr = _make_manager()
        vi = mgr.build_index(["alpha", "beta", "gamma"])
        result = mgr.similarity_search("test query", vi, k=2)
        assert isinstance(result, str)

    def test_similarity_search_empty_index(self):
        mgr = _make_manager()
        empty_vi = VectorIndex(index=MagicMock(), chunks=[])
        result = mgr.similarity_search("query", empty_vi)
        assert result == ""

    def test_similarity_search_k_capped(self):
        """k should be clamped to index size."""
        mgr = _make_manager()
        vi = mgr.build_index(["a", "b"])
        result = mgr.similarity_search("query", vi, k=100)  # should not crash
        assert isinstance(result, str)
