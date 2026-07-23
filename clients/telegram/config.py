"""
==============================================================================
Zenemoo AI - Telegram Bot Client Configuration
==============================================================================
Settings and environment variables specific to the Telegram Bot client interface.
"""

from pydantic_settings import BaseSettings, SettingsConfigDict


class TelegramBotSettings(BaseSettings):
    """Telegram Bot configuration settings."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    BOT_TOKEN: str = "8512332781:AAGhYn0O2iGC9sNABe3BqeqpCJv9NNJYzcQ"
    BACKEND_API_URL: str = "http://localhost:8000"
    API_TIMEOUT_SECONDS: int = 300
    MAX_RETRIES: int = 3
    RATE_LIMIT_PER_MIN: int = 15
    MAX_GPU_WORKERS: int = 3



bot_settings = TelegramBotSettings()
