"""
==============================================================================
Zenemoo AI - User Preferences & Settings Database Model
==============================================================================
SQLAlchemy ORM model storing user customized default enhancement settings.
"""

from datetime import datetime, timezone
from sqlalchemy import String, Integer, ForeignKey, DateTime
from sqlalchemy.orm import Mapped, mapped_column
from core.database import Base


class UserSettings(Base):
    """User default preferences model."""

    __tablename__ = "settings"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), unique=True, nullable=False)
    preferred_face_engine: Mapped[str] = mapped_column(String(32), default="gfpgan")
    default_scale: Mapped[int] = mapped_column(Integer, default=4)
    output_format: Mapped[str] = mapped_column(String(16), default="png")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )
