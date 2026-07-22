"""
Zenemoo AI Telegram Bot Module
Forwarding to decoupled Telegram Client in clients/telegram/
"""

from clients.telegram.bot_client import TelegramBotClient, start_bot

__all__ = ["TelegramBotClient", "start_bot"]
