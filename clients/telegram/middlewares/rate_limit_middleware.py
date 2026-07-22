"""
==============================================================================
Zenemoo AI - Telegram Bot Rate Limiting Middleware
==============================================================================
Per-user rate limit protection to prevent request flooding.
"""

import time
from typing import Dict
from telegram import Update
from telegram.ext import ContextTypes
from core.logging import logger
from clients.telegram.config import bot_settings


class RateLimiter:
    """In-memory sliding window rate limiter per user ID."""

    def __init__(self):
        self.user_requests: Dict[int, list] = {}
        self.limit = bot_settings.RATE_LIMIT_PER_MIN
        self.window = 60.0

    def is_allowed(self, user_id: int) -> bool:
        """Returns True if user is within rate limit window, False otherwise."""
        now = time.time()
        requests = self.user_requests.get(user_id, [])

        # Remove requests older than 60s
        requests = [t for t in requests if now - t < self.window]
        self.user_requests[user_id] = requests

        if len(requests) >= self.limit:
            return False

        self.user_requests[user_id].append(now)
        return True


rate_limiter = RateLimiter()


async def check_rate_limit(update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
    """Middleware checker for rate limiting."""
    if not update.effective_user:
        return True

    user_id = update.effective_user.id
    if not rate_limiter.is_allowed(user_id):
        logger.warning(f"⚠️ User {user_id} exceeded Telegram rate limit.")
        if update.message:
            await update.message.reply_text(
                "⚠️ **Rate Limit Exceeded**\n\nPlease wait a minute before sending another request."
            )
        return False
    return True
