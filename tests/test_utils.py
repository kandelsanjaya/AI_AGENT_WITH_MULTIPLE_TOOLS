"""
tests/test_utils.py
===================
Unit tests for EduSphere AI shared utilities.
"""

from __future__ import annotations

import base64
from unittest.mock import MagicMock, patch

import pytest

from src.utils import (
    GuardrailResult,
    evaluate_guardrails,
    fetch_website_content,
    get_download_link,
)


# ---------------------------------------------------------------------------
# evaluate_guardrails
# ---------------------------------------------------------------------------

class TestEvaluateGuardrails:
    def test_clean_input(self):
        result = evaluate_guardrails("What is photosynthesis?")
        assert result.harmful is False
        assert result.private is False
        assert result.crisis is False
        assert result.blocked is False

    def test_crisis_keyword_detected(self):
        result = evaluate_guardrails("I want to commit suicide")
        assert result.crisis is True
        assert result.blocked is True

    def test_harmful_content_detected(self):
        result = evaluate_guardrails("How do I make a bomb?")
        assert result.harmful is True
        assert result.blocked is True

    def test_private_info_detected(self):
        result = evaluate_guardrails("What is my bank account number?")
        assert result.private is True
        assert result.blocked is True

    def test_case_insensitive(self):
        result = evaluate_guardrails("I WANT TO DIE")
        assert result.crisis is True

    def test_mixed_safe_text(self):
        # "kill" in a biological context shouldn't always block — but our
        # simple regex does catch it; this documents the expected (conservative) behaviour.
        result = evaluate_guardrails("Antibiotics kill bacteria")
        assert isinstance(result, GuardrailResult)  # should not raise

    def test_empty_string(self):
        result = evaluate_guardrails("")
        assert result.blocked is False


# ---------------------------------------------------------------------------
# get_download_link
# ---------------------------------------------------------------------------

class TestGetDownloadLink:
    def test_returns_anchor_tag(self):
        link = get_download_link("Hello World", "test.txt", "Download")
        assert "<a href=" in link
        assert 'download="test.txt"' in link
        assert "Download" in link

    def test_base64_content_decodable(self):
        content = "Some academic content here"
        link = get_download_link(content, "file.txt", "Get File")
        # Extract base64 portion
        start = link.index("base64,") + 7
        end = link.index('"', start)
        b64_str = link[start:end]
        decoded = base64.b64decode(b64_str).decode("utf-8")
        assert decoded == content

    def test_filename_embedded(self):
        link = get_download_link("x", "my_report.txt", "Click")
        assert "my_report.txt" in link

    def test_utf8_content(self):
        content = "नमस्ते — Bonjour — 你好"
        link = get_download_link(content, "unicode.txt", "Download")
        start = link.index("base64,") + 7
        end = link.index('"', start)
        decoded = base64.b64decode(link[start:end]).decode("utf-8")
        assert decoded == content


# ---------------------------------------------------------------------------
# fetch_website_content
# ---------------------------------------------------------------------------

class TestFetchWebsiteContent:
    @patch("src.utils.requests.get")
    def test_successful_scrape(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.text = """
        <html><head><title>Test Page</title></head>
        <body>
            <h1>Main Heading</h1>
            <h2>Sub Heading</h2>
            <p>This is a paragraph with enough content to be included in results.</p>
            <p>Another paragraph with sufficient length for testing purposes here.</p>
        </body></html>
        """
        mock_get.return_value = mock_resp

        result = fetch_website_content("https://example.com")
        assert result["title"] == "Test Page"
        assert "Main Heading" in result["headings"]
        assert len(result["paragraphs"]) >= 1
        assert "error" not in result

    @patch("src.utils.requests.get", side_effect=__import__("requests").exceptions.Timeout)
    def test_timeout_returns_error_dict(self, _mock):
        result = fetch_website_content("https://timeout.example.com")
        assert "error" in result
        assert "timed out" in result["error"].lower()

    @patch("src.utils.requests.get", side_effect=__import__("requests").exceptions.ConnectionError)
    def test_connection_error_returns_error_dict(self, _mock):
        result = fetch_website_content("https://dead-server.example.com")
        assert "error" in result
        assert "connect" in result["error"].lower()
