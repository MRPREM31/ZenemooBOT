"""
==============================================================================
Zenemoo AI - Health & System Status Endpoints
==============================================================================
GET /health
GET /status
GET /status/{job_id}
"""

from fastapi import APIRouter, HTTPException, status
from core.config import settings
from shared.utils import get_gpu_info
from services import job_service
from api.schemas import HealthCheckResponse, JobStatusResponse

router = APIRouter(tags=["Health & Status"])


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Returns application health, environment status, and hardware GPU info."""
    gpu_info = get_gpu_info()
    return HealthCheckResponse(
        status="healthy",
        app_name=settings.APP_NAME,
        version="1.0.0",
        environment=settings.ENVIRONMENT,
        device=settings.compute_device,
        gpu_info=gpu_info,
    )


@router.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Retrieves async processing job status and progress percentage."""
    job = job_service.get_job_status(job_id)
    if not job:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job ID '{job_id}' not found."
        )
    return job
