"""
==============================================================================
Zenemoo AI - User Database Model
==============================================================================
SQLAlchemy ORM model for Telegram users, Web dashboard users, and administrators.
"""

from datetime import datetime, timezone
from typing import Optional, List
from sqlalchemy import String, Integer, Boolean, DateTime, BigInteger
from sqlalchemy.orm import Mapped, mapped_column, relationship
from core.database import Base


class User(Base):
    """User account entity."""

    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    telegram_id: Mapped[Optional[int]] = mapped_column(BigInteger, unique=True, index=True, nullable=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    email: Mapped[Optional[str]] = mapped_column(String(128), unique=True, index=True, nullable=True)
    hashed_password: Mapped[str] = mapped_column(String(256), nullable=False)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    is_admin: Mapped[bool] = mapped_column(Boolean, default=False)
    quota_limit: Mapped[int] = mapped_column(Integer, default=100)
    quota_used: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    images: Mapped[List["Image"]] = relationship("Image", back_populates="user", cascade="all, delete-orphan")
    jobs: Mapped[List["Job"]] = relationship("Job", back_populates="user", cascade="all, delete-orphan")
