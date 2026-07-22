"""
==============================================================================
Zenemoo AI - Authentication Request & Response Pydantic Schemas
==============================================================================
"""

from pydantic import BaseModel, EmailStr, Field, ConfigDict
from typing import Optional


class UserRegisterRequest(BaseModel):
    """User registration payload."""
    username: str = Field(..., min_length=3, max_length=32, json_schema_extra={"example": "john_doe"})
    password: str = Field(..., min_length=6, json_schema_extra={"example": "SecurePassword123!"})
    email: Optional[EmailStr] = Field(None, json_schema_extra={"example": "user@example.com"})
    telegram_id: Optional[int] = Field(None, json_schema_extra={"example": 123456789})


class UserResponse(BaseModel):
    """User profile response schema."""
    model_config = ConfigDict(from_attributes=True)

    id: int
    username: str
    email: Optional[str] = None
    telegram_id: Optional[int] = None
    is_active: bool
    is_admin: bool
    quota_limit: int
    quota_used: int


class TokenResponse(BaseModel):
    """JWT token response schema."""
    access_token: str
    token_type: str = "bearer"
    expires_in_minutes: int
    user: UserResponse
