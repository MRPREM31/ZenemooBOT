"""
==============================================================================
Zenemoo AI - Image Metadata Database Model
==============================================================================
SQLAlchemy ORM model storing uploaded image metadata and enhanced result links.
"""

from datetime import datetime, timezone
from typing import Optional
from sqlalchemy import String, Integer, ForeignKey, DateTime
from sqlalchemy.orm import Mapped, mapped_column, relationship
from core.database import Base


class Image(Base):
    """Image metadata entity."""

    __tablename__ = "images"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("users.id"), nullable=True, index=True)
    original_filename: Mapped[str] = mapped_column(String(256), nullable=False)
    stored_filename: Mapped[str] = mapped_column(String(256), nullable=False, unique=True)
    output_filename: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    file_size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    width: Mapped[int] = mapped_column(Integer, default=0)
    height: Mapped[int] = mapped_column(Integer, default=0)
    mime_type: Mapped[str] = mapped_column(String(64), default="image/jpeg")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

    # Relationships
    user: Mapped[Optional["User"]] = relationship("User", back_populates="images")
