"""
exceptions.py
=============
Custom exception hierarchy for DasaAI / EduSphere AI.
All application-specific errors should inherit from EduSphereError
so callers can catch them in a single except clause when needed.
"""

from __future__ import annotations


class EduSphereError(Exception):
    """Base exception for all EduSphere AI errors."""


class APIKeyMissingError(EduSphereError):
    """Raised when the GROQ_API_KEY environment variable is absent or empty."""

    def __init__(self) -> None:
        super().__init__(
            "GROQ_API_KEY is not configured. "
            "Please add it to your .env file and restart the application."
        )


class APIRequestError(EduSphereError):
    """Raised when an HTTP request to the GROQ API fails."""

    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(f"GROQ API error {status_code}: {detail}")


class DocumentProcessingError(EduSphereError):
    """Raised when a PDF or document cannot be parsed or embedded."""


class VectorStoreError(EduSphereError):
    """Raised when FAISS index operations fail."""


class GuardrailViolationError(EduSphereError):
    """Raised when user input triggers a safety or privacy guardrail."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(f"Guardrail violation: {reason}")


class AuthenticationError(EduSphereError):
    """Raised when credential verification fails."""
