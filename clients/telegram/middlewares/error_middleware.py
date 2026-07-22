"""
==============================================================================
Zenemoo AI - Telegram Bot Centralized Error Handler Middleware
==============================================================================
Intercepts uncaught bot exceptions, logs error stacktraces, and notifies users.
"""

from telegram import Update
from telegram.ext import ContextTypes
from core.logging import logger
from shared.exceptions.telegram_exception import TelegramBackendCommunicationException


async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Central error boundary capturing handler errors."""
    error = context.error
    logger.error(f"❌ Telegram Bot Error encountered: {error}", exc_info=error)

    if isinstance(update, Update) and update.effective_message:
        if isinstance(error, TelegramBackendCommunicationException):
            user_msg = (
                "⚠️ **Backend Service Unavailable**\n\n"
                "Could not connect to Zenemoo AI backend server. Please check backend API service status."
            )
        else:
            user_msg = (
                "⚠️ **An Unexpected Error Occurred**\n\n"
                "Our AI processing pipeline encountered an issue. Please try again shortly."
            )

        try:
            await update.effective_message.reply_text(user_msg, parse_mode="Markdown")
        except Exception as e:
            logger.error(f"Failed sending error message to user: {e}")
