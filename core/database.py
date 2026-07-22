"""
==============================================================================
Zenemoo AI - Database Module
==============================================================================
Provides SQLAlchemy 2.0 async engine configuration, session factories,
and database lifecycle management supporting SQLite and PostgreSQL.
"""

from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import (
    create_async_engine,
    async_sessionmaker,
    AsyncSession,
    AsyncEngine,
)
from sqlalchemy.orm import DeclarativeBase
from core.config import settings
from core.logging import logger


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy ORM models."""
    pass


# Build Async Engine
engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    future=True,
)

# Async Session Factory
AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def init_db() -> None:
    """Initializes database tables during application startup."""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info(f"Database initialized successfully using URL: {settings.DATABASE_URL}")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}", exc_info=True)
        raise e


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Dependency generator providing a transactional database session."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {e}")
            raise e
        finally:
            await session.close()
