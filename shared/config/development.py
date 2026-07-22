"""
Zenemoo AI - Development Environment Settings
"""

from pydantic_settings import BaseSettings


class DevelopmentConfig(BaseSettings):
    ENVIRONMENT: str = "development"
    DEBUG: bool = True
    LOG_LEVEL: str = "DEBUG"
    CORS_ORIGINS: list = ["*"]
    RELOAD: bool = True
