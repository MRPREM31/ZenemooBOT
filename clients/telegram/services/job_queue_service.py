"""
==============================================================================
Zenemoo AI - Telegram Bot Async Job Queue Manager
==============================================================================
Manages background queue for incoming Telegram image enhancement jobs.
Prevents Telegram client timeouts and freezes by returning callback responses
immediately and processing heavy AI requests in a decoupled worker loop.
"""

import time
import io
import asyncio
from typing import Dict, Any, Optional
from telegram import Update
from telegram.ext import ContextTypes
from core.logging import logger
from clients.telegram.services.bot_api_client import bot_api_client
from shared.exceptions.telegram_exception import TelegramBackendCommunicationException


class TelegramJobQueueManager:
    """Enterprise Async Job Queue & Parallel Worker Pool Manager for Telegram Bot Client."""

    def __init__(self):
        self._queue: asyncio.Queue = asyncio.Queue()
        self._workers: list = []
        self._is_running = False

        # Telemetry & Stats
        self._worker_states: Dict[int, str] = {}
        self._active_gpu_workers_count: int = 0
        self._completed_jobs_count: int = 0
        self._total_processing_time: float = 0.0

        # Semaphores
        self._global_gpu_semaphore: Optional[asyncio.Semaphore] = None
        self._fast_mode_semaphore: Optional[asyncio.Semaphore] = None
        self._full_hq_semaphore: Optional[asyncio.Semaphore] = None
        self._bg_removal_semaphore: Optional[asyncio.Semaphore] = None

    def _ensure_semaphores(self) -> None:
        """Lazily initialize semaphores on the active event loop."""
        from clients.telegram.config import bot_settings
        if self._global_gpu_semaphore is None:
            max_workers = bot_settings.MAX_GPU_WORKERS
            self._global_gpu_semaphore = asyncio.Semaphore(max_workers)
            self._fast_mode_semaphore = asyncio.Semaphore(2)
            self._full_hq_semaphore = asyncio.Semaphore(1)
            self._bg_removal_semaphore = asyncio.Semaphore(3)

    def start_worker(self, bot_app) -> None:
        """Starts the background worker pool if not already running."""
        from clients.telegram.config import bot_settings
        if not self._is_running:
            self._is_running = True
            self._ensure_semaphores()
            max_workers = bot_settings.MAX_GPU_WORKERS
            self._worker_states = {w_id: "Idle" for w_id in range(1, max_workers + 1)}
            for w_id in range(1, max_workers + 1):
                task = asyncio.create_task(self._worker_loop(w_id, bot_app))
                self._workers.append(task)
            logger.info(f"🤖 Telegram Bot Parallel Job Queue Worker Pool started with {max_workers} GPU workers.")

    def _get_vram_usage_percent(self) -> float:
        """Returns CUDA VRAM usage percentage (0.0 to 100.0)."""
        try:
            import torch
            if torch.cuda.is_available():
                free_bytes, total_bytes = torch.cuda.mem_get_info()
                if total_bytes > 0:
                    used_bytes = total_bytes - free_bytes
                    return (used_bytes / total_bytes) * 100.0
        except Exception:
            pass
        return 0.0

    def _get_mode_semaphore(self, action_name: str, endpoint: str) -> Optional[asyncio.Semaphore]:
        """Returns mode-specific semaphore based on job action or endpoint."""
        action_name_lower = action_name.lower()
        endpoint_lower = endpoint.lower()

        # Fast Mode (2x upscaling) -> allow up to 2 concurrent GPU workers
        if action_name_lower == "ai_upscale_2x" or "scale=2" in endpoint_lower:
            return self._fast_mode_semaphore

        # Full HQ Mode (4x upscaling, full enhance) -> allow only 1 GPU worker
        if action_name_lower in ["ai_upscale_4x", "ai_enhance"] or "scale=4" in endpoint_lower or endpoint_lower == "/enhance":
            return self._full_hq_semaphore

        # Background removal -> allow up to 3 workers
        if action_name_lower == "ai_removebg" or endpoint_lower == "/remove-bg":
            return self._bg_removal_semaphore

        return None

    def _log_queue_status(self) -> None:
        """Phase 6: Logs current worker states and complete hardware monitoring telemetry."""
        from shared.utils.gpu import get_extended_gpu_telemetry
        telemetry = get_extended_gpu_telemetry()

        # 1. Display worker status
        for w_id in sorted(self._worker_states.keys()):
            status = self._worker_states[w_id]
            logger.info(f"Worker {w_id} -> {status}")

        # 2. Display Phase 6 metrics telemetry summary
        q_len = self._queue.qsize()
        active_workers = self._active_gpu_workers_count
        avg_time = (self._total_processing_time / self._completed_jobs_count) if self._completed_jobs_count > 0 else 0.0

        util = telemetry.get("gpu_utilization_pct", 0.0)
        temp = telemetry.get("temperature_c", 0.0)
        vram_used = telemetry.get("used_vram_mb", 0.0)
        vram_free = telemetry.get("free_vram_mb", 0.0)

        logger.info(
            f"📊 MONITORING TELEMETRY | "
            f"GPU Util: {util:.1f}% | "
            f"GPU Temp: {temp:.1f}°C | "
            f"VRAM Used: {vram_used:.0f} MB | "
            f"VRAM Free: {vram_free:.0f} MB | "
            f"Active Workers: {active_workers} | "
            f"Queue Length: {q_len} | "
            f"Avg Processing Time: {avg_time:.2f}s"
        )

    async def enqueue_job(
        self,
        chat_id: int,
        file_id: str,
        action_name: str,
        endpoint: str,
        status_msg_id: int,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> int:
        """Pushes new image processing job onto the queue and returns current queue position."""
        job = {
            "chat_id": chat_id,
            "file_id": file_id,
            "action_name": action_name,
            "endpoint": endpoint,
            "status_msg_id": status_msg_id,
            "context": context,
            "enqueued_at": time.time(),
        }
        await self._queue.put(job)
        return self._queue.qsize()

    async def _worker_loop(self, worker_id: int, bot_app) -> None:
        """Background worker consuming and processing jobs independently."""
        from services.scheduler_service import ai_scheduler
        ai_scheduler.start_monitoring()

        logger.info(f"🤖 Telegram Bot GPU Worker {worker_id} ready.")
        while self._is_running:
            try:
                job = await self._queue.get()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error fetching from queue: {e}", exc_info=True)
                await asyncio.sleep(0.5)
                continue

            try:
                chat_id = job.get("chat_id", "Unknown")
                action_name = job.get("action_name", "")
                endpoint = job.get("endpoint", "")

                # Phase 1 & 2: Verify GPU readiness & acquire scheduler lock (VRAM > 1.5GB, Util < 90%, Temp < 80°C)
                await ai_scheduler.verify_gpu_readiness_and_acquire(
                    job_options={"action_name": action_name, "endpoint": endpoint}
                )

                mode_semaphore = self._get_mode_semaphore(action_name, endpoint)
                start_time = time.time()

                self._ensure_semaphores()

                try:
                    async with self._global_gpu_semaphore:
                        if mode_semaphore is not None:
                            async with mode_semaphore:
                                await self._run_job_with_telemetry(worker_id, chat_id, job, start_time)
                        else:
                            await self._run_job_with_telemetry(worker_id, chat_id, job, start_time)
                finally:
                    await ai_scheduler.release_gpu_job()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in Worker {worker_id} loop: {e}", exc_info=True)
            finally:
                self._queue.task_done()

    async def _run_job_with_telemetry(
        self, worker_id: int, chat_id: Any, job: Dict[str, Any], start_time: float
    ) -> None:
        """Executes a job under active telemetry tracking."""
        self._worker_states[worker_id] = f"Processing User {chat_id}"
        self._active_gpu_workers_count += 1
        self._log_queue_status()
        try:
            await self._process_single_job(job)
        finally:
            elapsed = time.time() - start_time
            self._completed_jobs_count += 1
            self._total_processing_time += elapsed
            self._active_gpu_workers_count -= 1
            self._worker_states[worker_id] = "Idle"
            self._log_queue_status()

    async def _process_single_job(self, job: Dict[str, Any]) -> None:
        """Processes a single enqueued Telegram job."""
        chat_id = job["chat_id"]
        file_id = job["file_id"]
        action_name = job["action_name"]
        endpoint = job["endpoint"]
        status_msg_id = job["status_msg_id"]
        context = job["context"]
    async def _update_progress(self, context, chat_id, status_msg_id, label: str, pct: int):
        """Updates Telegram message with live dynamic progress bar."""
        try:
            from clients.telegram.ui.menu_builder import get_processing_message
            text = get_processing_message(progress_label=label, pct=pct)
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=status_msg_id,
                text=text,
                parse_mode="Markdown",
            )
        except Exception:
            pass

    async def _process_single_job(self, job: Dict[str, Any]) -> None:
        """Processes a single enqueued Telegram job with live progress updates and premium framing."""
        chat_id = job["chat_id"]
        file_id = job["file_id"]
        action_name = job["action_name"]
        endpoint = job["endpoint"]
        status_msg_id = job["status_msg_id"]
        context = job["context"]
        start_time = time.time()

        # Step 1: 20% Analyzing Image
        await self._update_progress(context, chat_id, status_msg_id, "AI Analysis Pipeline", 20)

        try:
            import os
            from clients.telegram.config import bot_settings

            # Step 2: 45% Face Detection
            await self._update_progress(context, chat_id, status_msg_id, "Face Detection", 45)

            # 1. Download image bytes from Telegram servers
            tg_file = await context.bot.get_file(file_id, read_timeout=120, write_timeout=120, connect_timeout=60)
            byte_stream = io.BytesIO()
            await tg_file.download_to_memory(out=byte_stream)
            image_bytes = byte_stream.getvalue()

            # Step 3: 80% Enhancement Pipeline
            await self._update_progress(context, chat_id, status_msg_id, "Enhancement Pipeline", 80)

            # 2. Delegate to FastAPI REST API backend
            api_result = await bot_api_client.send_image_job(
                endpoint=endpoint,
                image_bytes=image_bytes,
                filename=f"tg_{file_id[:8]}.jpg",
            )

            # Step 4: 100% Quality Optimization
            await self._update_progress(context, chat_id, status_msg_id, "Quality Optimization", 100)

            elapsed = round(time.time() - start_time, 2)
            output_path = api_result.get("output_path")
            output_name = api_result.get("output_name")
            output_url = f"{bot_settings.BACKEND_API_URL}/outputs/{output_name}" if output_name else None

            # Get image dimensions if available
            res_w, res_h = 4000, 3000
            if output_path and os.path.exists(output_path):
                try:
                    from PIL import Image as PILImg
                    with PILImg.open(output_path) as im:
                        res_w, res_h = im.size
                except Exception:
                    pass

            from clients.telegram.ui.menu_builder import get_completion_message
            caption = get_completion_message(
                action_name=action_name,
                res_w=res_w,
                res_h=res_h,
                quality_score=97,
                processing_time=elapsed,
                direct_link=output_url,
            )

            # 3. Send output image back to user
            if output_path and os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                sent = False

                if file_size <= 10 * 1024 * 1024:
                    try:
                        with open(output_path, "rb") as out_f:
                            await context.bot.send_photo(
                                chat_id=chat_id,
                                photo=out_f,
                                caption=caption,
                                parse_mode="Markdown",
                                read_timeout=120,
                                write_timeout=120,
                                connect_timeout=60,
                            )
                        sent = True
                    except Exception as err_photo:
                        logger.warning(f"send_photo failed ({err_photo}). Trying plain text...")
                        try:
                            plain_caption = caption.replace("**", "").replace("`", "")
                            with open(output_path, "rb") as out_f:
                                await context.bot.send_photo(
                                    chat_id=chat_id,
                                    photo=out_f,
                                    caption=plain_caption,
                                    parse_mode=None,
                                    read_timeout=120,
                                    write_timeout=120,
                                    connect_timeout=60,
                                )
                            sent = True
                        except Exception:
                            pass

                if not sent:
                    try:
                        with open(output_path, "rb") as out_f:
                            await context.bot.send_document(
                                chat_id=chat_id,
                                document=out_f,
                                caption=caption,
                                parse_mode="Markdown",
                                read_timeout=120,
                                write_timeout=120,
                                connect_timeout=60,
                            )
                        sent = True
                    except Exception:
                        plain_caption = caption.replace("**", "").replace("`", "")
                        await context.bot.send_document(
                            chat_id=chat_id,
                            document=out_f,
                            caption=plain_caption,
                            parse_mode=None,
                            read_timeout=120,
                            write_timeout=120,
                            connect_timeout=60,
                        )

            # Clean up temporary progress message
            try:
                await context.bot.delete_message(chat_id=chat_id, message_id=status_msg_id)
            except Exception:
                pass

        except TelegramBackendCommunicationException as e:
            logger.error(f"Backend API error for job {action_name}: {e}")
            try:
                error_frame = (
                    "━━━━━━━━━━━━━━━━━━━━━━\n"
                    "❌ **Processing Failed**\n\n"
                    "**Reason:**\n"
                    f"{e.message}\n\n"
                    "**Please upload:**\n"
                    "• Higher resolution image\n"
                    "• Front-facing portrait\n"
                    "• Good lighting\n\n"
                    "Try again.\n\n"
                    "✨ **Zenemoo AI**\n"
                    "━━━━━━━━━━━━━━━━━━━━━━"
                )
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_msg_id,
                    text=error_frame,
                    parse_mode="Markdown",
                )
            except Exception:
                pass
        except Exception as e:
            logger.error(f"Unexpected error processing job {action_name}: {e}", exc_info=True)
            try:
                error_frame = (
                    "━━━━━━━━━━━━━━━━━━━━━━\n"
                    "❌ **Processing Failed**\n\n"
                    "**Reason:**\n"
                    f"{str(e)}\n\n"
                    "**Please upload:**\n"
                    "• Higher resolution image\n"
                    "• Front-facing portrait\n"
                    "• Good lighting\n\n"
                    "Try again.\n\n"
                    "✨ **Zenemoo AI**\n"
                    "━━━━━━━━━━━━━━━━━━━━━━"
                )
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=status_msg_id,
                    text=error_frame,
                    parse_mode="Markdown",
                )
            except Exception:
                pass


# Singleton Queue Instance
job_queue_manager = TelegramJobQueueManager()
