"""
database.py
===========
SQLite database layer for EduSphere AI.
Handles user accounts (register/login) and persistent chat history.
"""

from __future__ import annotations

import datetime
import hashlib
import logging
import secrets
import sqlite3
from contextlib import contextmanager
from typing import Optional

from .config import DB_PATH

log = logging.getLogger("edusphere.database")


def _hash_password(password: str, salt: str) -> str:
    """Hash a password with a salt using PBKDF2-SHA256 (100k iterations)."""
    return hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000
    ).hex()


@contextmanager
def get_connection():
    """Context manager for SQLite connections."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables if they do not already exist."""
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                full_name TEXT,
                salt TEXT NOT NULL,
                password_hash TEXT NOT NULL,
                theme TEXT DEFAULT 'dark',
                created_at TEXT NOT NULL
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS chats (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                session_id TEXT NOT NULL,
                session_title TEXT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
            """
        )
        c.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                title TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
    log.info("Database initialised at %s", DB_PATH)


# ---------------------- USER MANAGEMENT ---------------------- #

def register_user(
    username: str, email: str, password: str, full_name: str = ""
) -> tuple[bool, str]:
    """Register a new user with salted PBKDF2 hashed password."""
    username = username.strip().lower()
    email = email.strip().lower()
    if not username or not email or not password:
        return False, "All fields are required."

    salt = secrets.token_hex(16)
    pw_hash = _hash_password(password, salt)

    try:
        with get_connection() as conn:
            c = conn.cursor()
            c.execute(
                """INSERT INTO users (username, email, full_name, salt, password_hash, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    username,
                    email,
                    full_name,
                    salt,
                    pw_hash,
                    datetime.datetime.now().isoformat(),
                ),
            )
        log.info("New user registered: %s", username)
        return True, "Account created successfully! Please log in."
    except sqlite3.IntegrityError:
        return False, "Username or email already exists."


def authenticate_user(username_or_email: str, password: str) -> Optional[dict]:
    """Return user row dict if credentials are valid, else None."""
    identifier = username_or_email.strip().lower()
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            "SELECT * FROM users WHERE username = ? OR email = ?",
            (identifier, identifier),
        )
        row = c.fetchone()
        if row is None:
            return None
        expected_hash = _hash_password(password, row["salt"])
        if secrets.compare_digest(expected_hash, row["password_hash"]):
            log.info("Successful login for user: %s", identifier)
            return dict(row)
        return None


def update_user_theme(user_id: int, theme: str):
    """Persist a user's theme preference."""
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("UPDATE users SET theme = ? WHERE id = ?", (theme, user_id))


def get_user_by_id(user_id: int) -> Optional[dict]:
    """Fetch a user by their ID."""
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = c.fetchone()
        return dict(row) if row else None


# ---------------------- CHAT SESSIONS ---------------------- #

def create_session(user_id: int, session_id: str, title: str = "New Chat"):
    """Insert a new chat session (idempotent via INSERT OR IGNORE)."""
    now = datetime.datetime.now().isoformat()
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            """INSERT OR IGNORE INTO sessions (session_id, user_id, title, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?)""",
            (session_id, user_id, title, now, now),
        )


def update_session_title(session_id: str, title: str):
    """Update the title of a session."""
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            "UPDATE sessions SET title = ?, updated_at = ? WHERE session_id = ?",
            (title, datetime.datetime.now().isoformat(), session_id),
        )


def list_sessions(user_id: int) -> list[dict]:
    """List all sessions for a user, most recent first."""
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            "SELECT * FROM sessions WHERE user_id = ? ORDER BY updated_at DESC",
            (user_id,),
        )
        return [dict(r) for r in c.fetchall()]


def delete_session(session_id: str, user_id: int):
    """Delete a session and all its messages."""
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            "DELETE FROM chats WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        )
        c.execute(
            "DELETE FROM sessions WHERE session_id = ? AND user_id = ?",
            (session_id, user_id),
        )


def save_message(user_id: int, session_id: str, role: str, content: str):
    """Persist a single chat message."""
    now = datetime.datetime.now().isoformat()
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            """INSERT INTO chats (user_id, session_id, role, content, created_at)
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, session_id, role, content, now),
        )
        c.execute(
            "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
            (now, session_id),
        )


def get_session_messages(session_id: str, user_id: int) -> list[dict]:
    """Get all messages in a session, ordered chronologically."""
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            """SELECT role, content, created_at FROM chats
               WHERE session_id = ? AND user_id = ? ORDER BY id ASC""",
            (session_id, user_id),
        )
        return [dict(r) for r in c.fetchall()]


def get_all_user_messages(user_id: int) -> list[dict]:
    """Get all messages from all sessions for a user."""
    with get_connection() as conn:
        c = conn.cursor()
        c.execute(
            """SELECT session_id, role, content, created_at FROM chats
               WHERE user_id = ? ORDER BY id ASC""",
            (user_id,),
        )
        return [dict(r) for r in c.fetchall()]
