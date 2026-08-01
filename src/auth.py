"""
auth.py
=======
Authentication and user-management module for EduSphere AI.
Integrated with the SQLite database layer.

Security:
- Passwords are salted and hashed (using SQLite-backed PBKDF2-SHA256).
- The default accounts (student and admin) are seeded into SQLite if not present.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from .database import init_db, authenticate_user, register_user as db_register, get_connection
from .config import logger

log = logging.getLogger("edusphere.auth")

# Ensure DB is initialized (for tests as well as runtime)
init_db()


@dataclass
class UserRecord:
    """Represents a user record return type compatible with tests."""
    email: str
    name: str


def seed_default_accounts():
    """Seed the default demo accounts into SQLite database if they don't exist."""
    try:
        db_register(
            username="admin",
            email="admin@edusphere.ai",
            password="admin123",
            full_name="System Administrator",
            role="Executive Lead"
        )
    except Exception:
        pass

    try:
        db_register(
            username="student",
            email="student@edusphere.ai",
            password="student123",
            full_name="Alex Mercer",
            role="Graduate Scholar"
        )
    except Exception:
        pass


def verify_credentials(email: str, password: str) -> bool:
    """
    Verify *email* / *password* against the SQLite stored hash.
    """
    # Seed just in case
    seed_default_accounts()
    
    user = authenticate_user(email, password)
    return user is not None


def get_user_info(email: str) -> Optional[dict]:
    """
    Return a sanitised dict of user metadata.
    """
    email = email.strip().lower()
    from .database import get_connection
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email = ? OR username = ?", (email, email))
        row = c.fetchone()
        if row:
            return {
                "name": row["full_name"] or row["username"],
                "role": row["role"] or "Student",
                "email": row["email"],
                "username": row["username"],
                "id": row["id"]
            }
    return None


def register_user(
    email: str,
    password: str,
    name: str,
    role: str = "Student",
) -> UserRecord:
    """
    Register a new user in the SQLite database.
    """
    email = email.strip().lower()
    username = email.split("@")[0]
    
    success, msg = db_register(
        username=username,
        email=email,
        password=password,
        full_name=name,
        role=role
    )
    if not success:
        raise ValueError(f"Email '{email}' is already registered: {msg}")
        
    return UserRecord(email=email, name=name)


def update_user_credentials(
    current_email: str,
    new_email: Optional[str] = None,
    new_password: Optional[str] = None,
    new_name: Optional[str] = None,
) -> bool:
    """
    Update credentials or name for a registered user.
    """
    current_email = current_email.strip().lower()
    from .database import _hash_password
    import secrets
    
    with get_connection() as conn:
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE email = ?", (current_email,))
        row = c.fetchone()
        if not row:
            return False
        
        user_id = row["id"]
        
        if new_name:
            c.execute("UPDATE users SET full_name = ? WHERE id = ?", (new_name, user_id))
            
        if new_password:
            salt = secrets.token_hex(16)
            pw_hash = _hash_password(new_password, salt)
            c.execute("UPDATE users SET salt = ?, password_hash = ? WHERE id = ?", (salt, pw_hash, user_id))
            
        if new_email and new_email.strip().lower() != current_email:
            new_email = new_email.strip().lower()
            # Check uniqueness
            c.execute("SELECT id FROM users WHERE email = ? AND id != ?", (new_email, user_id))
            if c.fetchone():
                raise ValueError(f"Email '{new_email}' is already registered.")
            c.execute("UPDATE users SET email = ? WHERE id = ?", (new_email, user_id))
            
    log.info("User credentials updated successfully in SQLite.")
    return True
