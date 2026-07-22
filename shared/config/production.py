"""
Zenemoo AI - Production Environment Settings
"""

from pydantic_settings import BaseSettings


class ProductionConfig(BaseSettings):
    ENVIRONMENT: str = "production"
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    CORS_ORIGINS: list = ["https://zenemoo.ai"]
    RELOAD: bool = False
    WORKERS_COUNT: int = 4
