"""
==============================================================================
Zenemoo AI - Telegram Bot Client Unit Tests
==============================================================================
Tests Telegram Bot client configuration and handler setup.
"""

import pytest
from clients.telegram.config import bot_settings
from clients.telegram.bot_client import TelegramBotClient
from clients.telegram.services.bot_api_client import BackendAPIClient


def test_telegram_bot_client_init():
    """Tests Telegram Bot client initialization with token."""
    bot = TelegramBotClient(token="123456789:TEST_BOT_TOKEN")
    assert bot.token == "123456789:TEST_BOT_TOKEN"


def test_backend_api_client_url_formatting():
    """Tests REST API client base URL normalization."""
    api_client = BackendAPIClient(base_url="http://localhost:8000/")
    assert api_client.base_url == "http://localhost:8000"
