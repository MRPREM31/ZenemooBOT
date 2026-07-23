"""
==============================================================================
Zenemoo AI - Intelligent AI Scheduler & Dynamic Worker Scaling Service
==============================================================================
Orchestrates Phase 1 (Intelligent Scheduler admission control based on VRAM,
utilization %, and temperature) and Phase 2 (Dynamic Worker Scaling).
"""

import time
import asyncio
from typing import Dict, Any, Optional
from core.config import settings
from core.logging import logger
from shared.utils.gpu import get_extended_gpu_telemetry
from shared.utils.image_analyzer import analyze_image_properties, estimate_vram_requirement_gb


class IntelligentAIScheduler:
    """Enterprise Intelligent AI Scheduler & Dynamic Worker Scaling Engine."""

    def __init__(self):
        self._min_workers: int = 1
        self._max_workers_ceiling: int = 4
        self._current_allowed_workers: int = settings.MAX_GPU_WORKERS
        self._active_gpu_jobs: int = 0
        self._lock = asyncio.Lock()
        self._monitoring_task: Optional[asyncio.Task] = None
        self._is_monitoring: bool = False

    @property
    def current_allowed_workers(self) -> int:
        return self._current_allowed_workers

    @property
    def active_gpu_jobs(self) -> int:
        return self._active_gpu_jobs

    def start_monitoring(self) -> None:
        """Starts background hardware telemetry & dynamic worker scaling loop."""
        if not self._is_monitoring:
            self._is_monitoring = True
            self._monitoring_task = asyncio.create_task(self._dynamic_scaling_loop())
            logger.info("🧠 Intelligent AI Scheduler & Dynamic Worker Scaling active.")

    async def _dynamic_scaling_loop(self) -> None:
        """Phase 2: Continuously evaluates GPU hardware telemetry and scales worker capacity."""
        while self._is_monitoring:
            try:
                await asyncio.sleep(2.0)
                gpu_data = get_extended_gpu_telemetry()
                util = gpu_data.get("gpu_utilization_pct", 0.0)
                temp = gpu_data.get("temperature_c", 0.0)
                free_mb = gpu_data.get("free_vram_mb", 4000.0)

                async with self._lock:
                    prev_workers = self._current_allowed_workers

                    # Thermal Override: Temp > 80°C -> Scale down
                    if temp > 80.0:
                        self._current_allowed_workers = max(self._min_workers, self._current_allowed_workers - 1)
                        logger.warning(
                            f"🔥 GPU Temperature ({temp:.1f}°C > 80°C) high! Reducing workers: {prev_workers} -> {self._current_allowed_workers}"
                        )
                    # Utilization > 90% or Free VRAM < 1.5 GB -> Scale down
                    elif util > 90.0 or free_mb < 1500.0:
                        self._current_allowed_workers = max(self._min_workers, self._current_allowed_workers - 1)
                        if prev_workers != self._current_allowed_workers:
                            logger.info(
                                f"📉 GPU Load High (Util: {util:.1f}%, Free VRAM: {free_mb:.0f}MB). Scaling workers: {prev_workers} -> {self._current_allowed_workers}"
                            )
                    # Utilization < 50% and Temp < 75°C and Free VRAM > 2.5 GB -> Scale up
                    elif util < 50.0 and temp < 75.0 and free_mb > 2500.0:
                        self._current_allowed_workers = min(self._max_workers_ceiling, self._current_allowed_workers + 1)
                        if prev_workers != self._current_allowed_workers:
                            logger.info(
                                f"📈 GPU Utilization Low ({util:.1f}% < 50%, Temp: {temp:.1f}°C). Increasing workers: {prev_workers} -> {self._current_allowed_workers}"
                            )
                    # Utilization 70% - 90% -> Keep current workers
                    else:
                        pass
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in dynamic scaling loop: {e}")
                await asyncio.sleep(2.0)

    async def verify_gpu_readiness_and_acquire(
        self,
        image_input: Optional[bytes] = None,
        job_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Phase 1: Analyzes image, estimates VRAM requirement, checks hardware state,
        and waits until Free VRAM > 1.5 GB, GPU utilization < 90%, and Temp < 80°C.
        Acquires GPU job execution slot when conditions are met.
        """
        opts = job_options or {}

        # 1. Analyze Image & Estimate VRAM
        if image_input:
            analysis = analyze_image_properties(image_input)
            est_vram_gb = estimate_vram_requirement_gb(
                resolution_mp=analysis["megapixels"],
                face_restore=opts.get("face_restore", True),
                upscale_factor=int(opts.get("upscale_factor", 4)),
                remove_bg=opts.get("remove_bg", False),
                sharpen=opts.get("sharpen", True),
            )
        else:
            analysis = {"megapixels": 1.0, "has_faces": False}
            est_vram_gb = 1.5

        # 2. Admission Check Loop
        while True:
            gpu_data = get_extended_gpu_telemetry()
            free_mb = gpu_data.get("free_vram_mb", 4000.0)
            util = gpu_data.get("gpu_utilization_pct", 0.0)
            temp = gpu_data.get("temperature_c", 0.0)

            # Check capacity & thresholds
            has_worker_slot = self._active_gpu_jobs < self._current_allowed_workers
            vram_ok = free_mb >= 1500.0  # Free VRAM > 1.5 GB
            util_ok = util < 90.0         # GPU utilization < 90%
            temp_ok = temp < 80.0 if temp > 0 else True

            if has_worker_slot and vram_ok and util_ok and temp_ok:
                async with self._lock:
                    self._active_gpu_jobs += 1
                return {
                    "status": "acquired",
                    "analysis": analysis,
                    "estimated_vram_gb": est_vram_gb,
                    "telemetry": gpu_data,
                }

            # If overloaded or hot, wait in non-blocking loop
            reasons = []
            if not has_worker_slot:
                reasons.append(f"active jobs {self._active_gpu_jobs} >= limit {self._current_allowed_workers}")
            if not vram_ok:
                reasons.append(f"free VRAM {free_mb:.0f}MB < 1500MB")
            if not util_ok:
                reasons.append(f"utilization {util:.1f}% >= 90%")
            if not temp_ok:
                reasons.append(f"temp {temp:.1f}°C >= 80°C")

            logger.info(f"⏳ Scheduler Waiting for GPU capacity ({', '.join(reasons)})...")
            await asyncio.sleep(0.5)

    async def release_gpu_job(self) -> None:
        """Releases active GPU job execution slot."""
        async with self._lock:
            self._active_gpu_jobs = max(0, self._active_gpu_jobs - 1)


# Singleton Instance
ai_scheduler = IntelligentAIScheduler()
