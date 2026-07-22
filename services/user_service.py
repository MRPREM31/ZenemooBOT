"""
==============================================================================
Zenemoo AI - User Management Service Layer
==============================================================================
Manages user authentication, profile settings, rate limits, and processing quotas.
"""

import time
from typing import Dict, Any, Tuple, Optional
from core.logging import logger


class UserService:
    """User Management & Quota Enforcement Service with TTL memory management."""

    def __init__(self, ttl_seconds: int = 3600):
        # In-memory session tracking mapping telegram_id -> (used_count, last_updated_timestamp)
        self._user_quotas: Dict[int, Tuple[int, float]] = {}
        self.ttl_seconds = ttl_seconds

    def _cleanup_expired_quotas(self) -> None:
        """Evicts user quotas older than ttl_seconds."""
        now = time.time()
        expired = [uid for uid, (_, ts) in self._user_quotas.items() if now - ts > self.ttl_seconds]
        for uid in expired:
            self._user_quotas.pop(uid, None)

    def check_user_quota(self, telegram_id: int) -> bool:
        """Checks if user has remaining process quota for the current window."""
        self._cleanup_expired_quotas()
        quota_tuple = self._user_quotas.get(telegram_id)
        used = quota_tuple[0] if quota_tuple else 0
        return used < 100  # Default 100 free requests

    def increment_user_quota(self, telegram_id: int) -> int:
        """Increments usage counter for user."""
        self._cleanup_expired_quotas()
        quota_tuple = self._user_quotas.get(telegram_id)
        current_count = (quota_tuple[0] + 1) if quota_tuple else 1
        self._user_quotas[telegram_id] = (current_count, time.time())
        return current_count


# Singleton instance
user_service = UserService()

