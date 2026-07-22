"""
==============================================================================
Zenemoo AI - Telegram Bot Exception Domain
==============================================================================
Custom exceptions for Telegram Bot API communication, rate-limiting, and webhooks.
"""


class TelegramBotException(Exception):
    """Base exception for Telegram client and API interactions."""
    def __init__(self, message: str, chat_id: str = ""):
        super().__init__(message)
        self.message = message
        self.chat_id = chat_id


class TelegramRateLimitException(TelegramBotException):
    """Raised when Telegram API returns HTTP 429 Too Many Requests."""
    pass


class TelegramBackendCommunicationException(TelegramBotException):
    """Raised when Telegram Bot Client fails to connect to FastAPI Backend REST endpoints."""
    pass
