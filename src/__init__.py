"""
__init__.py
===========
EduSphere AI package marker. Exposes top-level symbols for convenience.
"""

from .config import THEMES, DEFAULT_THEME, logger
from .auth import verify_credentials, get_user_info
from .rag import VectorStoreManager, VectorIndex, extract_pdf_chunks, load_embedding_model
from .utils import evaluate_guardrails, get_download_link, fetch_website_content, groq_chat, stream_to_ui
from .exceptions import EduSphereError, APIKeyMissingError, GuardrailViolationError

__all__ = [
    "THEMES",
    "DEFAULT_THEME",
    "logger",
    "verify_credentials",
    "get_user_info",
    "VectorStoreManager",
    "VectorIndex",
    "extract_pdf_chunks",
    "load_embedding_model",
    "evaluate_guardrails",
    "get_download_link",
    "fetch_website_content",
    "groq_chat",
    "stream_to_ui",
    "EduSphereError",
    "APIKeyMissingError",
    "GuardrailViolationError",
]
