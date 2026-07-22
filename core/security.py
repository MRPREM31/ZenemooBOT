"""
==============================================================================
Zenemoo AI - Security & Authentication Utilities
==============================================================================
Provides cryptographic functions for JWT token encoding/decoding,
password hashing using PassLib (Bcrypt/Argon2), and request token verification.
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Any, Dict
from jose import jwt, JWTError
from passlib.context import CryptContext
from core.config import settings
from core.logging import logger

# Password Hashing Context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verifies a plain-text password against its hashed value."""
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Generates a secure hash of the provided plain-text password."""
    return pwd_context.hash(password)


def create_access_token(
    data: Dict[str, Any],
    expires_delta: Optional[timedelta] = None,
) -> str:
    """Encodes a JWT access token with user payload claims and expiration."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode,
        settings.SECRET_KEY,
        algorithm=settings.ALGORITHM,
    )
    return encoded_jwt


def decode_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Decodes and validates a JWT access token. Returns payload dict or None."""
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[settings.ALGORITHM],
        )
        return payload
    except JWTError as e:
        logger.warning(f"JWT Token validation failed: {e}")
        return None
