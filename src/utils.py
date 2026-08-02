"""
utils.py
========
Shared helper utilities for EduSphere AI.

Covers:
- Guardrail evaluation (safety, privacy, crisis detection)
- Download link generation (text, PDF, image, audio)
- Web page scraping / content extraction
- GROQ API streaming and non-streaming calls
- LICT Campus knowledge search & greeting detection
- Interaction logging
- Chat export (Markdown, JSON, PDF)
"""

from __future__ import annotations

import base64
import datetime
import io
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, List, Optional

import requests
import streamlit as st
from bs4 import BeautifulSoup

from .config import (
    CRISIS_KEYWORDS,
    CRISIS_RESOURCES,
    DATA_JSON,
    DEFAULT_MODEL,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    HARMFUL_PATTERNS,
    MAX_INTERACTION_LOG,
    PRIVATE_PATTERNS,
)
from .exceptions import APIKeyMissingError, APIRequestError

log = logging.getLogger("edusphere.utils")

# ---------------------------------------------------------------------------
# Guardrails
# ---------------------------------------------------------------------------

@dataclass
class GuardrailResult:
    """Outcome of a guardrail evaluation pass."""

    harmful: bool = False
    private: bool = False
    crisis: bool = False

    @property
    def blocked(self) -> bool:
        return self.harmful or self.private or self.crisis


def evaluate_guardrails(text: str) -> GuardrailResult:
    """
    Evaluate *text* against safety, privacy, and crisis patterns.

    Args:
        text: Raw user input string.

    Returns:
        GuardrailResult indicating which (if any) flags were triggered.
    """
    lowered = text.lower()
    return GuardrailResult(
        harmful=any(re.search(p, lowered) for p in HARMFUL_PATTERNS),
        private=any(re.search(p, lowered) for p in PRIVATE_PATTERNS),
        crisis=any(kw in lowered for kw in CRISIS_KEYWORDS),
    )


# ---------------------------------------------------------------------------
# Download Helpers
# ---------------------------------------------------------------------------

def get_download_link(content: str, filename: str, label: str) -> str:
    """
    Generate an HTML anchor tag that downloads *content* as a text file.

    Args:
        content: Text content to embed.
        filename: Suggested download filename.
        label: Human-readable link label.

    Returns:
        HTML string with a data-URI download link.
    """
    b64 = base64.b64encode(content.encode("utf-8")).decode("utf-8")
    return (
        f'<a href="data:text/plain;charset=utf-8;base64,{b64}" '
        f'download="{filename}" class="download-btn">⬇️ {label}</a>'
    )


def get_pdf_download_link(content: str, filename: str, label: str) -> str:
    """Generate a PDF download link from text content using fpdf2."""
    try:
        from fpdf import FPDF

        pdf = FPDF()
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.add_page()
        pdf.set_font("Helvetica", size=11)

        # Sanitize text to avoid Latin-1 encoding errors
        content_clean = (
            content.replace("—", "-")
            .replace("–", "-")
            .replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
            .replace("™", "TM")
            .replace("©", "(c)")
            .replace("®", "(r)")
            .replace("●", "*")
            .replace("⚡", "*")
            .replace("🏫", "")
            .replace("🎯", "")
            .replace("📋", "")
            .replace("📊", "")
            .replace("🔑", "")
        )
        content_clean = content_clean.encode("latin-1", "replace").decode("latin-1")

        # Handle multi-line content with markdown-like headers
        for line in content_clean.split("\n"):
            stripped = line.strip()
            if stripped.startswith("###"):
                pdf.set_font("Helvetica", "B", 13)
                pdf.multi_cell(0, 8, stripped.lstrip("#").strip())
                pdf.set_font("Helvetica", size=11)
            elif stripped.startswith("##"):
                pdf.set_font("Helvetica", "B", 14)
                pdf.multi_cell(0, 9, stripped.lstrip("#").strip())
                pdf.set_font("Helvetica", size=11)
            elif stripped.startswith("#"):
                pdf.set_font("Helvetica", "B", 16)
                pdf.multi_cell(0, 10, stripped.lstrip("#").strip())
                pdf.set_font("Helvetica", size=11)
            elif stripped.startswith("**") and stripped.endswith("**"):
                pdf.set_font("Helvetica", "B", 11)
                pdf.multi_cell(0, 6, stripped.strip("*").strip())
                pdf.set_font("Helvetica", size=11)
            elif stripped.startswith("- ") or stripped.startswith("* "):
                pdf.multi_cell(0, 6, "     • " + stripped[2:])
            elif stripped == "":
                pdf.ln(4)
            else:
                # Clean markdown formatting for PDF
                clean = re.sub(r"\*\*(.*?)\*\*", r"\1", stripped)
                clean = re.sub(r"\*(.*?)\*", r"\1", clean)
                clean = re.sub(r"`(.*?)`", r"\1", clean)
                pdf.multi_cell(0, 6, clean)

        pdf_bytes = bytes(pdf.output())
        b64 = base64.b64encode(pdf_bytes).decode("utf-8")
        return (
            f'<a href="data:application/pdf;base64,{b64}" '
            f'download="{filename}" class="download-btn">📄 {label}</a>'
        )
    except ImportError:
        log.warning("fpdf2 not installed — falling back to text download")
        return get_download_link(content, filename.replace(".pdf", ".txt"), label)
    except Exception as exc:
        log.error("PDF generation failed: %s", exc)
        return get_download_link(content, filename.replace(".pdf", ".txt"), label)


def get_audio_download_link(text: str, filename: str = "speech.mp3", label: str = "Download MP3") -> str:
    """Generate an MP3 audio file from text using gTTS and return a download link."""
    try:
        from gtts import gTTS

        tts = gTTS(text=text[:5000], lang="en")  # Cap at 5000 chars
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return (
            f'<a href="data:audio/mpeg;base64,{b64}" '
            f'download="{filename}" class="download-btn">🔊 {label}</a>'
        )
    except ImportError:
        log.warning("gTTS not installed — audio download unavailable")
        return ""
    except Exception as exc:
        log.error("TTS generation failed: %s", exc)
        return ""


def generate_pdf_bytes(content: str) -> bytes:
    """Generate and return styled PDF bytes from markdown text content using fpdf2."""
    try:
        from fpdf import FPDF

        class StyledPDF(FPDF):
            def header(self):
                # Accent vertical sidebar
                self.set_fill_color(59, 130, 246) # Blue accent
                self.rect(0, 0, 7, 297, "F")
                
                # Top header bar
                self.set_fill_color(30, 41, 59) # Slate background
                self.rect(7, 0, 203, 14, "F")
                
                # Header text
                self.set_font("Helvetica", "B", 7.5)
                self.set_text_color(226, 232, 240)
                self.text(12, 8, "EDUSPHERE AI ACADEMIC SYSTEM - VERIFIED REPORT")
                self.set_text_color(0, 0, 0) # Reset

            def footer(self):
                # Footer text
                self.set_y(-12)
                self.set_font("Helvetica", "I", 7.5)
                self.set_text_color(100, 116, 139)
                self.text(130, 290, f"Page {self.page_no()} | Compiled by EduSphere AI Suite")

        pdf = StyledPDF()
        pdf.set_left_margin(14)
        pdf.set_font("Helvetica", size=10) # Set default font before adding page
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.alias_nb_pages()
        pdf.add_page()
        
        # Start spacing
        pdf.ln(12)

        # Sanitize text to avoid Latin-1 encoding errors (replace curly quotes, em-dashes, etc.)
        content_clean = (
            content.replace("—", "-")
            .replace("–", "-")
            .replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
            .replace("™", "TM")
            .replace("©", "(c)")
            .replace("®", "(r)")
            .replace("●", "*")
            .replace("⚡", "*")
            .replace("🏫", "")
            .replace("🎯", "")
            .replace("📋", "")
            .replace("📊", "")
            .replace("🔑", "")
        )
        # Encode and decode back using latin-1 ignore to clean out any remaining unsupported characters
        content_clean = content_clean.encode("latin-1", "replace").decode("latin-1")

        # Handle multi-line content with markdown-like headers
        for line in content_clean.split("\n"):
            pdf.set_x(14) # Reset X coordinate before every line to avoid drift
            stripped = line.strip()
            
            # Draw dividers
            if stripped.startswith("===") or stripped.startswith("---"):
                pdf.set_draw_color(203, 213, 225)
                pdf.set_line_width(0.4)
                pdf.line(14, pdf.get_y() + 2, 200, pdf.get_y() + 2)
                pdf.set_x(14)
                pdf.ln(5)
                continue

            if stripped.startswith("###"):
                pdf.ln(2)
                pdf.set_font("Helvetica", "B", 12.5)
                pdf.set_text_color(30, 41, 59) # Slate
                pdf.multi_cell(0, 7, stripped.lstrip("#").strip())
                pdf.set_font("Helvetica", size=10)
                pdf.set_text_color(0, 0, 0)
                pdf.ln(1)
            elif stripped.startswith("##"):
                pdf.ln(3)
                pdf.set_font("Helvetica", "B", 13.5)
                pdf.set_text_color(37, 99, 235) # Accent Blue
                pdf.multi_cell(0, 8, stripped.lstrip("#").strip())
                pdf.set_font("Helvetica", size=10)
                pdf.set_text_color(0, 0, 0)
                pdf.ln(2)
            elif stripped.startswith("#"):
                pdf.ln(4)
                pdf.set_font("Helvetica", "B", 16)
                pdf.set_text_color(17, 24, 39) # Deep Gray
                pdf.multi_cell(0, 9, stripped.lstrip("#").strip())
                pdf.set_font("Helvetica", size=10)
                pdf.set_text_color(0, 0, 0)
                pdf.ln(3)
            elif stripped.startswith("**") and stripped.endswith("**"):
                pdf.set_font("Helvetica", "B", 10.5)
                pdf.multi_cell(0, 6, stripped.strip("*").strip())
                pdf.set_font("Helvetica", size=10)
            elif stripped.startswith("- ") or stripped.startswith("* "):
                pdf.set_font("Helvetica", size=10)
                pdf.multi_cell(0, 5.5, "     * " + stripped[2:])
            elif stripped == "":
                pdf.ln(3)
            else:
                pdf.set_font("Helvetica", size=10)
                # Clean markdown formatting for PDF
                clean = re.sub(r"\*\*(.*?)\*\*", r"\1", stripped)
                clean = re.sub(r"\*(.*?)\*", r"\1", clean)
                clean = re.sub(r"`(.*?)`", r"\1", clean)
                pdf.multi_cell(0, 5.5, clean)

        return bytes(pdf.output())
    except Exception as exc:
        log.error("PDF generation failed: %s", exc)
        # Final fallback: encode content using latin-1 to avoid download failures
        return content.encode("latin-1", "replace")


def generate_audio_bytes(text: str) -> Optional[bytes]:
    """Generate and return MP3 audio bytes using gTTS."""
    try:
        from gtts import gTTS
        tts = gTTS(text=text[:5000], lang="en")
        buf = io.BytesIO()
        tts.write_to_fp(buf)
        return buf.getvalue()
    except Exception as exc:
        log.error("Audio generation failed: %s", exc)
        return None



# ---------------------------------------------------------------------------
# Web Scraper
# ---------------------------------------------------------------------------

def fetch_website_content(url: str, timeout: int = 10) -> dict:
    """
    Scrape *url* and return structured content (title, headings, paragraphs).

    Args:
        url: HTTP/HTTPS URL to fetch.
        timeout: Request timeout in seconds.

    Returns:
        Dict with keys ``title``, ``headings``, ``paragraphs``, ``url``.
        On failure, returns dict with key ``error``.
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36"
        )
    }
    try:
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "aside"]):
            tag.decompose()

        title = soup.title.get_text(strip=True) if soup.title else "No Title"
        headings = [h.get_text(strip=True) for h in soup.find_all(["h1", "h2", "h3"])[:10]]
        paragraphs = [
            p.get_text(strip=True)
            for p in soup.find_all("p")
            if len(p.get_text(strip=True)) > 30
        ][:8]

        log.info("Scraped URL: %s | headings=%d paragraphs=%d", url, len(headings), len(paragraphs))
        return {"title": title, "headings": headings, "paragraphs": paragraphs, "url": url}

    except requests.exceptions.Timeout:
        log.warning("Timeout fetching URL: %s", url)
        return {"error": "Request timed out. The server took too long to respond.", "url": url}
    except requests.exceptions.ConnectionError:
        log.warning("Connection error fetching URL: %s", url)
        return {"error": "Could not connect to the server. Check the URL.", "url": url}
    except Exception as exc:  # noqa: BLE001
        log.error("Unexpected error scraping %s: %s", url, exc)
        return {"error": str(exc), "url": url}


# ---------------------------------------------------------------------------
# GROQ API Client
# ---------------------------------------------------------------------------

def _get_api_key() -> str:
    """Re-read GROQ_API_KEY at call time so env hot-reload works, prioritizing custom session key."""
    import streamlit as st
    try:
        if "custom_groq_api_key" in st.session_state and st.session_state.custom_groq_api_key.strip():
            return st.session_state.custom_groq_api_key.strip()
    except Exception:
        pass
    key = os.environ.get("GROQ_API_KEY", "") or GROQ_API_KEY
    if not key:
        raise APIKeyMissingError()
    return key


def _get_model(provided_model: str) -> str:
    """Resolve the active model, prioritizing custom session key choices when default is used."""
    import streamlit as st
    if provided_model == DEFAULT_MODEL:
        try:
            if "custom_groq_model" in st.session_state and st.session_state.custom_groq_model:
                return st.session_state.custom_groq_model
            elif "dashboard_model" in st.session_state and st.session_state.dashboard_model:
                return st.session_state.dashboard_model
        except Exception:
            pass
    return provided_model


def _build_headers() -> dict:
    return {
        "Authorization": f"Bearer {_get_api_key()}",
        "Content-Type": "application/json",
    }


def groq_request(
    messages: List[dict],
    model: str = DEFAULT_MODEL,
    stream: bool = False,
    max_tokens: int = 2048,
    system: Optional[str] = None,
    temperature: float = 0.7,
) -> requests.Response:
    """
    Send a chat completion request to the GROQ API.

    Args:
        messages: List of ``{"role": ..., "content": ...}`` dicts.
        model: GROQ model identifier.
        stream: Whether to enable SSE streaming.
        max_tokens: Maximum output tokens.
        system: Optional system prompt (prepended automatically).
        temperature: Sampling temperature.

    Returns:
        A :class:`requests.Response` object (not yet consumed).

    Raises:
        APIKeyMissingError: When GROQ_API_KEY is absent.
        APIRequestError: On non-200 HTTP responses.
    """
    model = _get_model(model)
    all_messages = []
    if system:
        all_messages.append({"role": "system", "content": system})
    all_messages.extend(messages)

    payload = {
        "model": model,
        "messages": all_messages,
        "max_tokens": max_tokens,
        "stream": stream,
        "temperature": temperature,
    }

    try:
        resp = requests.post(
            f"{GROQ_BASE_URL}/chat/completions",
            json=payload,
            headers=_build_headers(),
            stream=stream,
            timeout=60,
        )
        if not resp.ok:
            if resp.status_code == 401:
                raise APIRequestError(
                    401,
                    "Invalid or expired GROQ API key. "
                    "Please update your GROQ API key in the Settings & Profile page "
                    "or check the .env file.",
                )
            raise APIRequestError(resp.status_code, resp.text[:300])
        return resp
    except APIKeyMissingError:
        raise
    except APIRequestError:
        raise
    except requests.exceptions.Timeout:
        raise APIRequestError(408, "Request to GROQ API timed out.")
    except requests.exceptions.ConnectionError as exc:
        raise APIRequestError(503, f"Connection error: {exc}") from exc


def groq_stream(
    prompt: str,
    system: str = "You are an expert AI tutor.",
    model: str = DEFAULT_MODEL,
) -> Generator[str, None, None]:
    """
    Stream tokens from the GROQ API one chunk at a time.

    Yields:
        Individual text delta strings.
    """
    try:
        resp = groq_request(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            system=system,
            stream=True,
        )
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line.startswith("data: "):
                continue
            chunk_str = line[6:]
            if chunk_str.strip() == "[DONE]":
                break
            try:
                obj = json.loads(chunk_str)
                delta = obj["choices"][0]["delta"].get("content", "")
                if delta:
                    yield delta
            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    except (APIKeyMissingError, APIRequestError) as exc:
        yield f"\n\n⚠️ **API Error:** {exc}"
    except Exception as exc:  # noqa: BLE001
        log.error("Unexpected streaming error: %s", exc)
        yield f"\n\n⚠️ **Unexpected error:** {exc}"


def groq_chat(
    prompt: str,
    system: str = "You are an expert educational AI.",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
) -> str:
    """
    Non-streaming chat completion.

    Returns:
        The assistant reply as a plain string.
        On failure, returns a user-friendly error message.
    """
    try:
        resp = groq_request(
            messages=[{"role": "user", "content": prompt}],
            model=model,
            system=system,
            stream=False,
            max_tokens=max_tokens,
        )
        return resp.json()["choices"][0]["message"]["content"]
    except (APIKeyMissingError, APIRequestError) as exc:
        log.error("GROQ chat error: %s", exc)
        return f"⚠️ **API Error:** {exc}"
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        log.error("Malformed GROQ response: %s", exc)
        return "⚠️ Received an unexpected response from the API. Please try again."
    except Exception as exc:  # noqa: BLE001
        log.error("Unexpected GROQ error: %s", exc)
        return f"⚠️ **Unexpected error:** {exc}"


def groq_chat_with_history(
    messages: List[dict],
    system: str = "You are an expert educational AI.",
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
    temperature: float = 0.7,
) -> str:
    """
    Non-streaming chat completion with full message history.

    Args:
        messages: List of role/content message dicts (including prior history).
        system: System prompt.
        model: GROQ model identifier.
        max_tokens: Maximum output tokens.
        temperature: Sampling temperature.

    Returns:
        The assistant reply as a plain string.
    """
    try:
        resp = groq_request(
            messages=messages,
            model=model,
            system=system,
            stream=False,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.json()["choices"][0]["message"]["content"]
    except (APIKeyMissingError, APIRequestError) as exc:
        log.error("GROQ chat error: %s", exc)
        return f"⚠️ **API Error:** {exc}"
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        log.error("Malformed GROQ response: %s", exc)
        return "⚠️ Received an unexpected response from the API. Please try again."
    except Exception as exc:  # noqa: BLE001
        log.error("Unexpected GROQ error: %s", exc)
        return f"⚠️ **Unexpected error:** {exc}"


def stream_to_ui(
    prompt: str,
    system: str = "You are an expert AI tutor.",
    model: str = DEFAULT_MODEL,
) -> str:
    """
    Stream a GROQ response directly into a Streamlit placeholder.

    Args:
        prompt: User message.
        system: System prompt.
        model: Model identifier.

    Returns:
        The fully assembled response string.
    """
    placeholder = st.empty()
    full_response = ""
    for token in groq_stream(prompt, system=system, model=model):
        full_response += token
        placeholder.markdown(full_response + "▌")
        time.sleep(0.005)
    placeholder.markdown(full_response)
    return full_response


def duckduckgo_search(query: str, max_results: int = 3) -> List[dict]:
    """
    Perform a live web search using DuckDuckGo HTML search and extract results.
    Returns a list of dicts containing title, snippet, and link.
    """
    url = f"https://html.duckduckgo.com/html/?q={requests.utils.quote(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    }
    try:
        resp = requests.get(url, headers=headers, timeout=8)
        if not resp.ok:
            log.warning("DuckDuckGo search returned status %d", resp.status_code)
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for a in soup.find_all("a", class_="result__snippet")[:max_results]:
            parent = a.find_parent("div", class_="result")
            if not parent:
                continue
            title_elem = parent.find("a", class_="result__url")
            if title_elem:
                title = title_elem.get_text(strip=True)
                snippet = a.get_text(strip=True)
                link = title_elem.get("href", "")

                # Clean up redirect link if present
                if "uddg=" in link:
                    from urllib.parse import parse_qs, urlparse
                    parsed = urlparse(link)
                    qs = parse_qs(parsed.query)
                    link = qs.get("uddg", [link])[0]

                results.append({"title": title, "snippet": snippet, "link": link})
        log.info("DuckDuckGo search query '%s' yielded %d results", query, len(results))
        return results
    except Exception as exc:
        log.error("DuckDuckGo web search scraper failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# LICT Campus Greeting Detection
# ---------------------------------------------------------------------------

# Mapping of normalised greetings (including common typos) → data.json key
_GREETING_MAP: dict[str, str] = {
    "hi": "hi",
    "hii": "hii",
    "hiii": "hii",
    "hello": "hello",
    "helo": "hello",
    "helloo": "hello",
    "hey": "hey",
    "heya": "hey",
    "heyy": "hey",
    "good morning": "good morning",
    "morning": "good morning",
    "good afternoon": "good afternoon",
    "afternoon": "good afternoon",
    "good evening": "good evening",
    "evening": "good evening",
    "namaste": "namaste",
    "namaskar": "namaste",
    "bye": "bye",
    "goodbye": "bye",
    "good bye": "bye",
    "see you": "bye",
    "thanks": "thanks",
    "thank you": "thank you",
    "thankyou": "thank you",
    "thx": "thanks",
    "ty": "thanks",
}


def detect_greeting(text: str) -> Optional[str]:
    """
    Check if *text* is a greeting. Returns the data.json greeting key
    (e.g. 'hi', 'hello', 'namaste') or None if not a greeting.
    """
    cleaned = re.sub(r"[^\w\s]", "", text.strip().lower())
    cleaned = " ".join(cleaned.split())  # normalise whitespace
    return _GREETING_MAP.get(cleaned)


def get_greeting_response(key: str) -> str:
    """Return a random canned greeting response from data.json."""
    try:
        with open(DATA_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        greetings = data.get("greetings", {})
        options = greetings.get(key, [])
        if options:
            return random.choice(options)
    except Exception as exc:
        log.error("Failed to read greetings from data.json: %s", exc)
    return "Hello! How can I help you today? 😊"


# ---------------------------------------------------------------------------
# College Knowledge Search
# ---------------------------------------------------------------------------

def _load_data_json() -> dict:
    """Load and return data.json content."""
    try:
        with open(DATA_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        log.error("Failed to load data.json: %s", exc)
        return {}


def search_lict_knowledge(query: str) -> str:
    """
    Search the College knowledge base in data.json for relevant information.
    Returns a context string if relevant info is found, empty string otherwise.
    """
    data = _load_data_json()
    if not data:
        return ""

    query_lower = query.lower()
    context_parts = []

    kb = data.get("knowledge_base", {})
    faq = data.get("faq", [])

    # Check FAQ first — these are the most targeted answers
    for item in faq:
        q_words = set(item["question"].lower().split())
        query_words = set(query_lower.split())
        overlap = q_words & query_words
        # If at least 2 meaningful words match, include the answer
        noise = {"is", "the", "a", "an", "of", "what", "who", "where", "how", "does", "can", "do"}
        meaningful_overlap = overlap - noise
        if len(meaningful_overlap) >= 1:
            context_parts.append(f"Q: {item['question']}\nA: {item['answer']}")

    # Check institute info keywords
    institute = kb.get("institute", {})
    lict_keywords = [
        "lict", "lumbini", "ict", "campus", "gaindakot", "nawalpur",
        "college", "contact", "phone", "email", "address", "location", "where",
    ]
    if any(kw in query_lower for kw in lict_keywords):
        context_parts.append(
            f"Institute: {institute.get('name', '')}\n"
            f"Address: {institute.get('address', '')}\n"
            f"Phone: {', '.join(institute.get('phone', []))}\n"
            f"Email: {institute.get('email', '')}\n"
            f"Website: {institute.get('website', '')}"
        )

    # Check for course-related queries
    course_keywords = ["course", "program", "bsc", "csit", "bim", "bca", "bhm",
                       "bachelor", "study", "offer", "admission"]
    if any(kw in query_lower for kw in course_keywords):
        courses = kb.get("courses_offered", [])
        if courses:
            course_text = "\n".join(
                f"- {c['code']}: {c['full_name']} ({c.get('url', '')})"
                for c in courses
            )
            context_parts.append(f"Courses offered:\n{course_text}")

    # Check for leadership queries
    leader_keywords = ["principal", "chairman", "head", "chief", "koirala",
                       "pratap", "kailash", "leader", "management"]
    if any(kw in query_lower for kw in leader_keywords):
        leadership = kb.get("leadership", {})
        for role_key, person in leadership.items():
            context_parts.append(
                f"{person.get('title', role_key)}: {person.get('name', '')}\n"
                f"Message: {person.get('message', '')}"
            )

    # About/vision/mission
    about_keywords = ["about", "vision", "mission", "commitment", "history"]
    if any(kw in query_lower for kw in about_keywords):
        for key in ["about", "vision", "mission", "commitment"]:
            val = kb.get(key, "")
            if val:
                context_parts.append(f"{key.title()}: {val}")

    if not context_parts:
        return ""

    return "\n\n".join(context_parts)


# ---------------------------------------------------------------------------
# Interaction Logging
# ---------------------------------------------------------------------------

def log_interaction_to_json(user: str, role: str, content: str):
    """Append a message to the interaction_log in data.json (rolling cap)."""
    try:
        data = _load_data_json()
        log_entries = data.get("interaction_log", [])
        log_entries.append({
            "user": user,
            "role": role,
            "content": content[:500],  # Cap content length
            "timestamp": datetime.datetime.now().isoformat(),
        })
        # Keep only the last MAX_INTERACTION_LOG entries
        if len(log_entries) > MAX_INTERACTION_LOG:
            log_entries = log_entries[-MAX_INTERACTION_LOG:]
        data["interaction_log"] = log_entries
        with open(DATA_JSON, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
    except Exception as exc:
        log.error("Failed to log interaction: %s", exc)


# ---------------------------------------------------------------------------
# Chat Export
# ---------------------------------------------------------------------------

def export_chat_as_markdown(messages: list[dict], title: str = "EduSphere AI Chat Export") -> str:
    """Convert a message list to a downloadable Markdown string."""
    lines = [f"# {title}", f"*Exported: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M')}*", ""]
    for msg in messages:
        role = "👤 **You**" if msg.get("role") == "user" else "🎓 **Assistant**"
        lines.append(f"{role}:")
        lines.append(msg.get("content", msg.get("msg", "")))
        lines.append("")
    return "\n".join(lines)


def export_chat_as_json(messages: list[dict]) -> str:
    """Convert a message list to a downloadable JSON string."""
    export_data = []
    for msg in messages:
        export_data.append({
            "role": msg.get("role", ""),
            "content": msg.get("content", msg.get("msg", "")),
            "time": msg.get("time", msg.get("created_at", "")),
        })
    return json.dumps(export_data, indent=2, ensure_ascii=False)


from contextlib import contextmanager

@contextmanager
def logo_spinner(text: str):
    """A context manager that displays a premium custom logo spinner with a 0-100% progress line."""
    placeholder = st.empty()
    placeholder.markdown(
        f"""
        <div class="logo-loader-container">
            <div class="logo-loader-ring">
                <div class="logo-loader-inner">🎓</div>
            </div>
            <div class="logo-loader-text">{text}</div>
            <div class="logo-loader-progress-bg">
                <div class="logo-loader-progress-bar"></div>
            </div>
            <div class="logo-loader-percent">Processing... 0% to 100%</div>
        </div>
        """,
        unsafe_allow_html=True
    )
    try:
        yield
    finally:
        placeholder.empty()


