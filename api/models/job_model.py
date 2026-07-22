"""
==============================================================================
Zenemoo AI - Job Processing Database Model
==============================================================================
SQLAlchemy ORM model tracking pipeline execution jobs, options, and duration.
"""

from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, Integer, Float, ForeignKey, DateTime, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from core.database import Base


class Job(Base):
    """Job tracking entity."""

    __tablename__ = "jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    job_id: Mapped[str] = mapped_column(String(64), unique=True, index=True, nullable=False)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    image_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("images.id"), nullable=True, index=True)
    job_type: Mapped[str] = mapped_column(String(64), nullable=False)  # full_enhance, removebg, upscale, etc.
    status: Mapped[str] = mapped_column(String(32), default="pending")  # pending, processing, completed, failed
    progress: Mapped[int] = mapped_column(Integer, default=0)
    duration_ms: Mapped[float] = mapped_column(Float, default=0.0)
    options_json: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_log: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )

    # Relationships
    user: Mapped[Optional["User"]] = relationship("User", back_populates="jobs")
