"""
tests/test_auth.py
==================
Unit tests for the EduSphere AI authentication module.
"""

from __future__ import annotations

import pytest

from src.auth import get_user_info, register_user, verify_credentials


# ---------------------------------------------------------------------------
# verify_credentials
# ---------------------------------------------------------------------------

class TestVerifyCredentials:
    def test_valid_student_credentials(self):
        assert verify_credentials("student@edusphere.ai", "student123") is True

    def test_valid_admin_credentials(self):
        assert verify_credentials("admin@edusphere.ai", "admin123") is True

    def test_wrong_password(self):
        assert verify_credentials("student@edusphere.ai", "wrongpassword") is False

    def test_unknown_email(self):
        assert verify_credentials("nobody@nowhere.com", "anything") is False

    def test_empty_password(self):
        assert verify_credentials("student@edusphere.ai", "") is False

    def test_empty_email(self):
        assert verify_credentials("", "student123") is False

    def test_case_sensitivity_email(self):
        # Email lookup is case-insensitive (strip + lower applied)
        assert verify_credentials("STUDENT@EDUSPHERE.AI", "student123") is True

    def test_sql_injection_attempt(self):
        # Should not raise; simply returns False
        assert verify_credentials("' OR '1'='1", "' OR '1'='1") is False


# ---------------------------------------------------------------------------
# get_user_info
# ---------------------------------------------------------------------------

class TestGetUserInfo:
    def test_returns_user_dict(self):
        info = get_user_info("student@edusphere.ai")
        assert info is not None
        assert info["name"] == "Alex Mercer"
        assert info["role"] == "Graduate Scholar"
        assert "password_hash" not in info  # must not expose hash

    def test_unknown_user_returns_none(self):
        assert get_user_info("unknown@example.com") is None

    def test_case_insensitive_lookup(self):
        info = get_user_info("ADMIN@EDUSPHERE.AI")
        assert info is not None
        assert info["role"] == "Executive Lead"


# ---------------------------------------------------------------------------
# register_user
# ---------------------------------------------------------------------------

class TestRegisterUser:
    def test_new_user_registration(self):
        record = register_user(
            email="newuser@test.com",
            password="secure_pass_99",
            name="New User",
            role="Student",
        )
        assert record.email == "newuser@test.com"
        assert record.name == "New User"

    def test_registered_user_can_login(self):
        register_user(
            email="logintest@test.com",
            password="pass1234",
            name="Login Tester",
        )
        assert verify_credentials("logintest@test.com", "pass1234") is True

    def test_duplicate_registration_raises(self):
        register_user(
            email="dup@test.com",
            password="abc",
            name="Duplicate",
        )
        with pytest.raises(ValueError, match="already registered"):
            register_user(email="dup@test.com", password="xyz", name="Dup2")
