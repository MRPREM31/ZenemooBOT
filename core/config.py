"""
==============================================================================
Zenemoo AI - Core Settings & Configuration Module
==============================================================================
Manages environment variables, directory paths, hardware acceleration (GPU/CPU)
auto-detection, and application defaults using Pydantic Settings.
"""

import os
from pathlib import Path
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global system configuration model validated via Pydantic."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Metadata
    APP_NAME: str = "Zenemoo AI"
    ENVIRONMENT: str = "development"
    DEBUG: bool = True

    # Telegram Bot Settings
    BOT_TOKEN: str = ""

    # Backend API Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    SECRET_KEY: str = "zenemoo-secret-key-change-in-production-super-secure-token"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440

    # Database Configuration
    DATABASE_URL: str = "sqlite+aiosqlite:///./zenemoo.db"

    # Directory Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    UPLOAD_DIR: Path = Path("uploads")
    OUTPUT_DIR: Path = Path("outputs")
    TEMP_DIR: Path = Path("temp")
    STATIC_DIR: Path = Path("static")

    # Security & Processing Limits
    MAX_UPLOAD_SIZE_MB: int = 25
    RATE_LIMIT_PER_MINUTE: int = 20

    # Hardware & Performance
    DEVICE: str = "auto"
    NUM_WORKERS: int = 4
    MAX_CONCURRENT_GPU_JOBS: int = 2
    JOB_CACHE_TTL_SECONDS: int = 3600

    @property
    def compute_device(self) -> str:
        """Dynamically detect and return PyTorch compute device ('cuda' or 'cpu')."""
        if self.DEVICE.lower() in ["cuda", "gpu"]:
            return "cuda"
        elif self.DEVICE.lower() == "cpu":
            return "cpu"
        else:
            # Auto-detect CUDA availability
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"

    def ensure_directories(self) -> None:
        """Create storage directories if they do not exist."""
        for path_attr in [self.UPLOAD_DIR, self.OUTPUT_DIR, self.TEMP_DIR, self.STATIC_DIR]:
            abs_path = self.BASE_DIR / path_attr if not path_attr.is_absolute() else path_attr
            abs_path.mkdir(parents=True, exist_ok=True)


# Instantiated global singleton settings object
settings = Settings()
settings.ensure_directories()
