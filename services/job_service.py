"""
==============================================================================
Zenemoo AI - Job Queue Service Layer
==============================================================================
Manages async job state, processing progress tracking, and job lifecycle.
"""

import uuid
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from core.logging import logger


class JobService:
    """Async Job State & Queue Service with bounded memory management."""

    def __init__(self, max_jobs: int = 1000, ttl_seconds: int = 3600):
        self._jobs: Dict[str, Dict[str, Any]] = {}
        self.max_jobs = max_jobs
        self.ttl_seconds = ttl_seconds

    def _cleanup_expired_jobs(self) -> None:
        """Evicts expired jobs and enforces max memory limit."""
        now = datetime.now(timezone.utc)
        expired_ids = []
        for j_id, j in self._jobs.items():
            try:
                updated_at = datetime.fromisoformat(j["updated_at"])
                if (now - updated_at).total_seconds() > self.ttl_seconds:
                    expired_ids.append(j_id)
            except Exception:
                pass

        for j_id in expired_ids:
            self._jobs.pop(j_id, None)

        if len(self._jobs) > self.max_jobs:
            excess = len(self._jobs) - self.max_jobs
            for _ in range(excess):
                self._jobs.pop(next(iter(self._jobs)), None)

    def create_job(self, job_type: str, user_id: int) -> str:
        """Creates a new job entry and returns a unique job_id."""
        self._cleanup_expired_jobs()
        job_id = uuid.uuid4().hex
        self._jobs[job_id] = {
            "job_id": job_id,
            "job_type": job_type,
            "user_id": user_id,
            "status": "pending",
            "progress": 0,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "result": None,
            "error": None,
        }
        logger.info(f"📋 Created new job '{job_id}' (type: {job_type}, user: {user_id})")
        return job_id

    def update_job_progress(self, job_id: str, progress: int, status: str = "processing") -> None:
        """Updates progress percentage and status of a running job."""
        if job_id in self._jobs:
            self._jobs[job_id]["progress"] = progress
            self._jobs[job_id]["status"] = status
            self._jobs[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()

    def complete_job(self, job_id: str, result: Dict[str, Any]) -> None:
        """Marks a job as completed with result payload."""
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = "completed"
            self._jobs[job_id]["progress"] = 100
            self._jobs[job_id]["result"] = result
            self._jobs[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
            logger.info(f"✅ Completed job '{job_id}'")

    def fail_job(self, job_id: str, error_message: str) -> None:
        """Marks a job as failed with error details."""
        if job_id in self._jobs:
            self._jobs[job_id]["status"] = "failed"
            self._jobs[job_id]["error"] = error_message
            self._jobs[job_id]["updated_at"] = datetime.now(timezone.utc).isoformat()
            logger.error(f"❌ Failed job '{job_id}': {error_message}")

    def get_job_status(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Returns current state of a job."""
        self._cleanup_expired_jobs()
        return self._jobs.get(job_id)


# Singleton instance
job_service = JobService()

