"""
config.py
=========
Central configuration for EduSphere AI / LICT Campus Assistant.
Handles environment variables, constants, logging setup, and UI theme definitions.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR: Path = Path(__file__).resolve().parents[1]
LOG_DIR: Path = ROOT_DIR / "logs"
ASSETS_DIR: Path = ROOT_DIR / "assets"
DATA_JSON: Path = ROOT_DIR / "data.json"

import sys
if "pytest" in sys.modules:
    DB_PATH: str = str(ROOT_DIR / "test_users.db")
    # Clean up test DB file at startup
    test_db_path = Path(DB_PATH)
    if test_db_path.exists():
        try:
            test_db_path.unlink()
        except Exception:
            pass
else:
    DB_PATH: str = str(ROOT_DIR / "users.db")

LOG_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure application-level logging to file + console."""
    _logger = logging.getLogger("edusphere")
    if _logger.handlers:
        return _logger  # already configured

    _logger.setLevel(level)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    _logger.addHandler(ch)

    fh = logging.FileHandler(LOG_DIR / "app.log", encoding="utf-8")
    fh.setFormatter(formatter)
    _logger.addHandler(fh)

    return _logger


logger: logging.Logger = setup_logging()

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------
_ENV_PATH = ROOT_DIR / ".env"
load_dotenv(dotenv_path=_ENV_PATH, override=False)

GROQ_API_KEY: str = os.getenv("GROQ_API_KEY") or os.getenv("GROK_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
GROQ_BASE_URL: str = "https://api.groq.com/openai/v1"

# ---------------------------------------------------------------------------
# LLM Models
# ---------------------------------------------------------------------------
DEFAULT_MODEL: str = "llama-3.1-8b-instant"
AVAILABLE_MODELS: list[str] = [
    "llama-3.1-8b-instant",
    "llama-3.1-70b-versatile",
    "llama3-8b-8192",
    "llama3-70b-8192",
    "mixtral-8x7b-32768",
    "gemma2-9b-it",
]

# ---------------------------------------------------------------------------
# UI Themes — 6 premium themes
# ---------------------------------------------------------------------------
THEMES: dict[str, dict[str, str]] = {
    "✨ Futuristic Glassmorphism": {
        "bg": "#0b1120",
        "card": "rgba(255, 255, 255, 0.08)",
        "accent": "#3b82f6",
        "accent2": "#8b5cf6",
        "text": "#f8fafc",
        "sub": "#94a3b8",
        "glow": "rgba(59, 130, 246, 0.45)",
        "border_neon": "#06b6d4",
    },
    "🛸 Cyber Command Center": {
        "bg": "#0d1517",
        "card": "#142225",
        "accent": "#00ff66",
        "accent2": "#00aa55",
        "text": "#d1f4e6",
        "sub": "#008844",
        "glow": "rgba(0, 255, 102, 0.4)",
        "border_neon": "#00ff66",
    },
    "💼 Minimal AI SaaS": {
        "bg": "#f8fafc",
        "card": "#ffffff",
        "accent": "#3b82f6",
        "accent2": "#1d4ed8",
        "text": "#0f172a",
        "sub": "#64748b",
        "glow": "rgba(59, 130, 246, 0.15)",
        "border_neon": "#3b82f6",
    },
    "🌌 Holographic Neon": {
        "bg": "#050505",
        "card": "#0c0c14",
        "accent": "#8b5cf6",
        "accent2": "#d946ef",
        "text": "#f3e8ff",
        "sub": "#a855f7",
        "glow": "rgba(139, 92, 246, 0.55)",
        "border_neon": "#3b82f6",
    },
    "🎓 College Dark": {
        "bg": "#08090d",
        "card": "#14161e",
        "accent": "#00f0ff",
        "accent2": "#8b5cf6",
        "text": "#eef1f8",
        "sub": "#8b93a7",
        "glow": "rgba(0, 240, 255, 0.4)",
        "border_neon": "#00f0ff",
    },
    "☀️ College Light": {
        "bg": "#f4f6fb",
        "card": "#ffffff",
        "accent": "#0091ff",
        "accent2": "#7c3aed",
        "text": "#14161f",
        "sub": "#636b7e",
        "glow": "rgba(0, 145, 255, 0.35)",
        "border_neon": "#0091ff",
    },
}

DEFAULT_THEME: str = "✨ Futuristic Glassmorphism"

# ---------------------------------------------------------------------------
# Safety / Guardrail Patterns
# ---------------------------------------------------------------------------
HARMFUL_PATTERNS: list[str] = [
    r"\b(kill|murder|suicide|self.harm|harm myself|end my life|want to die|how to die)\b",
    r"\b(bomb|explosive|weapon|gun|knife to hurt)\b",
    r"\b(hack|crack password|steal data|bypass security)\b",
    r"\b(drug|cocaine|heroin|meth|illegal substance)\b",
    r"\b(porn|nude|naked|sexual content)\b",
    r"\b(racist|slur|hate speech)\b",
]

PRIVATE_PATTERNS: list[str] = [
    r"\b(password|api key|secret|credential)\b",
    r"\b(bank|account|card number|salary|income)\b",
]

CRISIS_KEYWORDS: list[str] = [
    "suicide",
    "kill myself",
    "end my life",
    "want to die",
    "self harm",
    "hurt myself",
]

CRISIS_RESOURCES: str = """
🆘 **Crisis Support Resources Available:**
- **Nepal:** TPO Nepal Helpline — 01-4460084
- **International:** Befrienders Worldwide — www.befrienders.org
- **Crisis Text Line:** Text HOME to 741741 (US/UK/Canada)
- **WHO Support:** www.who.int/mental_health

You are not alone. Please consider reaching out to qualified support services.
"""

# ---------------------------------------------------------------------------
# RAG / Embedding
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
RAG_CHUNK_SIZE: int = 800
RAG_CHUNK_OVERLAP: int = 150
RAG_TOP_K: int = 3

# ---------------------------------------------------------------------------
# Interaction log cap
# ---------------------------------------------------------------------------
MAX_INTERACTION_LOG: int = 500
