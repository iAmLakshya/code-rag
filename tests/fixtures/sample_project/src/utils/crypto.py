"""Cryptographic utilities for password hashing and verification."""

import hashlib
import secrets
from typing import Tuple


SALT_LENGTH = 32
HASH_ITERATIONS = 100000


def generate_salt() -> str:
    """Generate a random salt for password hashing.

    Returns:
        Hex-encoded salt string.
    """
    return secrets.token_hex(SALT_LENGTH)


def hash_password(password: str, salt: str = None) -> str:
    """Hash a password using PBKDF2.

    Args:
        password: Plain text password to hash.
        salt: Optional salt. Generated if not provided.

    Returns:
        Combined salt and hash string.
    """
    if salt is None:
        salt = generate_salt()

    password_hash = hashlib.pbkdf2_hmac(
        'sha256',
        password.encode('utf-8'),
        salt.encode('utf-8'),
        HASH_ITERATIONS
    ).hex()

    return f"{salt}${password_hash}"


def verify_hash(password: str, stored_hash: str) -> bool:
    """Verify a password against a stored hash.

    Args:
        password: Plain text password to verify.
        stored_hash: Previously stored hash string.

    Returns:
        True if password matches, False otherwise.
    """
    try:
        salt, expected_hash = stored_hash.split('$')
        computed_hash = hash_password(password, salt)
        return secrets.compare_digest(computed_hash, stored_hash)
    except ValueError:
        return False


def generate_token(length: int = 32) -> str:
    """Generate a secure random token.

    Args:
        length: Length of the token in bytes.

    Returns:
        URL-safe token string.
    """
    return secrets.token_urlsafe(length)
