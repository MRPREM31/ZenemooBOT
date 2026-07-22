"""
==============================================================================
Zenemoo AI - Web Admin Dashboard Stats Router
==============================================================================
GET /admin/stats
Provides real-time system health metrics, hardware GPU/VRAM telemetry,
storage disk breakdown, registered user metrics, and processing job audit logs.
"""

import os
import psutil
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from core.database import get_db
from core.config import settings
from shared.utils import get_gpu_info
from api.models import User, Image, Job, History

router = APIRouter(prefix="/admin", tags=["Admin Dashboard"])


def _calculate_dir_size_mb(directory_path) -> float:
    """Calculates total size of files in a directory in Megabytes."""
    total_size = 0
    abs_path = settings.BASE_DIR / directory_path if not directory_path.is_absolute() else directory_path
    if abs_path.exists():
        for dirpath, _, filenames in os.walk(abs_path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                total_size += os.path.getsize(fp)
    return round(total_size / (1024 * 1024), 2)


@router.get("/stats")
async def get_admin_dashboard_stats(db: AsyncSession = Depends(get_db)):
    """Returns comprehensive system metrics for the Web Admin Dashboard."""
    # 1. User & Image Counts
    user_count_stmt = select(func.count(User.id))
    image_count_stmt = select(func.count(Image.id))
    job_count_stmt = select(func.count(Job.id))

    users_count = (await db.execute(user_count_stmt)).scalar_one() or 0
    images_count = (await db.execute(image_count_stmt)).scalar_one() or 0
    jobs_count = (await db.execute(job_count_stmt)).scalar_one() or 0

    # 2. Hardware Telemetry via psutil & PyTorch
    cpu_percent = psutil.cpu_percent(interval=None)
    ram = psutil.virtual_memory()
    disk = psutil.disk_usage(str(settings.BASE_DIR))

    gpu_telemetry = get_gpu_info()

    # 3. Storage Usage Breakdown
    upload_size_mb = _calculate_dir_size_mb(settings.UPLOAD_DIR)
    output_size_mb = _calculate_dir_size_mb(settings.OUTPUT_DIR)
    temp_size_mb = _calculate_dir_size_mb(settings.TEMP_DIR)
    total_storage_mb = round(upload_size_mb + output_size_mb + temp_size_mb, 2)

    # 4. Recent Jobs Log
    recent_jobs_stmt = select(Job).order_by(Job.id.desc()).limit(10)
    recent_jobs_res = await db.execute(recent_jobs_stmt)
    recent_jobs = [
        {
            "job_id": j.job_id,
            "job_type": j.job_type,
            "status": j.status,
            "progress": j.progress,
            "duration_ms": j.duration_ms,
            "created_at": j.created_at.isoformat() if j.created_at else "",
        }
        for j in recent_jobs_res.scalars().all()
    ]

    return {
        "status": "online",
        "app_name": settings.APP_NAME,
        "environment": settings.ENVIRONMENT,
        "metrics": {
            "total_users": users_count,
            "total_images_processed": images_count,
            "total_jobs": jobs_count,
            "avg_latency_ms": 845.0,  # Simulated / benchmark average
        },
        "system": {
            "cpu_percent": cpu_percent,
            "ram_percent": ram.percent,
            "ram_used_gb": round(ram.used / (1024**3), 2),
            "ram_total_gb": round(ram.total / (1024**3), 2),
            "disk_percent": disk.percent,
            "disk_free_gb": round(disk.free / (1024**3), 2),
            "disk_total_gb": round(disk.total / (1024**3), 2),
        },
        "gpu": gpu_telemetry,
        "storage": {
            "upload_mb": upload_size_mb,
            "output_mb": output_size_mb,
            "temp_mb": temp_size_mb,
            "total_mb": total_storage_mb,
        },
        "recent_jobs": recent_jobs,
    }
