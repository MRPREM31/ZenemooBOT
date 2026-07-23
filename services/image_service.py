"""
==============================================================================
Zenemoo AI - Image Processing Service Layer
==============================================================================
Façade service layer coordinating image validation, storage, AI pipeline
enhancements, and metrics tracking. All API endpoints and Telegram Bot commands
use this service layer instead of directly invoking AI models.
"""

import io
import asyncio
from pathlib import Path
from typing import Dict, Any, Optional
from PIL import Image
from core.config import settings
from core.logging import logger
from services.storage_service import storage_service
from shared.utils.validators import validate_image_upload
from shared.utils.timer import time_execution
from shared.exceptions.image_exception import ImageProcessingException


class ImageService:
    """Enterprise Service façade for image processing operations."""

    def __init__(self):
        self.storage = storage_service
        self._gpu_semaphore: Optional[asyncio.Semaphore] = None

    @property
    def gpu_semaphore(self) -> asyncio.Semaphore:
        """Lazy-loaded asyncio.Semaphore instance on active event loop."""
        if self._gpu_semaphore is None:
            self._gpu_semaphore = asyncio.Semaphore(settings.MAX_GPU_WORKERS)
        return self._gpu_semaphore

    async def process_image_enhancement(
        self,
        file_bytes: bytes,
        filename: str,
        pipeline_options: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Validates, saves upload, delegates processing through AI pipeline,
        and saves enhanced output image. Returns complete metadata dictionary.
        """
        options = pipeline_options or {}
        
        # 1. CPU Stage 1: Validate Upload & Save Upload Bytes (outside GPU lock)
        validate_image_upload(file_bytes, filename)
        upload_name, upload_path = self.storage.save_upload_bytes(file_bytes, filename)

        # 2. Phase 1 & 2: Intelligent Scheduler GPU Readiness Admission Check
        from services.scheduler_service import ai_scheduler
        await ai_scheduler.verify_gpu_readiness_and_acquire(
            image_input=file_bytes,
            job_options=options,
        )

        # 3. GPU Critical Section (PyTorch model forward passes)
        try:
            with time_execution("AI_Pipeline_Execution") as timer:
                logger.info(f"⚡ Delegating image '{upload_name}' to AI processing pipeline (options: {options})...")
                
                mode = options.get("mode", "full_enhance")

                def _sync_process() -> bytes:
                    if mode == "full_enhance":
                        from ai.enhancer import unified_pipeline
                        return unified_pipeline.run_pipeline(
                            image_bytes=file_bytes,
                            filename=filename,
                            options=options,
                        )
                    elif mode == "background_removal":
                        from ai.background import bg_remover_engine
                        from ai.compress import compress_engine
                        result_pil = bg_remover_engine.remove_background(file_bytes)
                        opt_bytes, _ = compress_engine.optimize_output(result_pil, target_format="PNG")
                        return opt_bytes
                    elif mode == "face_restore":
                        from ai.restore import face_restorer_manager
                        from ai.compress import compress_engine
                        model_choice = options.get("model", "gfpgan")
                        fidelity = float(options.get("fidelity", 0.7))
                        input_pil = Image.open(io.BytesIO(file_bytes))
                        result_pil = face_restorer_manager.restore_face(
                            image_input=input_pil,
                            model=model_choice,
                            fidelity=fidelity,
                        )
                        opt_bytes, _ = compress_engine.optimize_output(result_pil, target_format="JPEG", jpeg_quality=95)
                        return opt_bytes
                    elif mode == "upscale":
                        from ai.upscale import realesrgan_engine
                        from ai.compress import compress_engine
                        scale = int(options.get("scale", 4))
                        input_pil = Image.open(io.BytesIO(file_bytes))
                        result_pil = realesrgan_engine.upscale(input_pil, scale=scale)
                        opt_bytes, _ = compress_engine.optimize_output(result_pil)
                        return opt_bytes
                    elif mode == "sharpen":
                        from ai.sharpen import sharpen_engine
                        from ai.compress import compress_engine
                        input_pil = Image.open(io.BytesIO(file_bytes))
                        result_pil = sharpen_engine.sharpen_and_denoise(input_pil)
                        opt_bytes, _ = compress_engine.optimize_output(result_pil, target_format="JPEG", jpeg_quality=95)
                        return opt_bytes
                    elif mode == "colorize":
                        from ai.colorize import colorize_engine
                        from ai.compress import compress_engine
                        render_factor = int(options.get("render_factor", 35))
                        vintage_mode = bool(options.get("vintage_mode", False))
                        input_pil = Image.open(io.BytesIO(file_bytes))
                        result_pil = colorize_engine.colorize(input_pil, render_factor=render_factor, vintage_mode=vintage_mode)
                        opt_bytes, _ = compress_engine.optimize_output(result_pil, target_format="JPEG", jpeg_quality=95)
                        return opt_bytes
                    elif mode == "compress":
                        from ai.compress import compress_engine
                        quality = int(options.get("quality", 95))
                        fmt = options.get("output_format")
                        opt_bytes, _ = compress_engine.optimize_output(file_bytes, target_format=fmt, jpeg_quality=quality)
                        return opt_bytes
                    elif mode == "passport":
                        from ai.passport import passport_engine
                        from ai.compress import compress_engine
                        country = options.get("country", "india")
                        bg_color = options.get("bg_color", "white")
                        input_pil = Image.open(io.BytesIO(file_bytes))
                        result_pil, _ = passport_engine.generate_passport(input_pil, country=country, bg_color_name=bg_color)
                        opt_bytes, _ = compress_engine.optimize_output(result_pil, target_format="PNG")
                        return opt_bytes
                    elif mode == "night":
                        from ai.night import night_engine
                        from ai.compress import compress_engine
                        input_pil = Image.open(io.BytesIO(file_bytes))
                        result_pil, _ = night_engine.enhance_night_photo(input_pil)
                        opt_bytes, _ = compress_engine.optimize_output(result_pil, target_format="JPEG", jpeg_quality=95)
                        return opt_bytes
                    elif mode == "portrait_studio":
                        from ai.portrait import portrait_engine
                        from ai.compress import compress_engine
                        portrait_mode = options.get("portrait_mode", "linkedin")
                        input_pil = Image.open(io.BytesIO(file_bytes))
                        result_pil, _ = portrait_engine.enhance_portrait(input_pil, mode=portrait_mode)
                        opt_bytes, _ = compress_engine.optimize_output(result_pil, target_format="JPEG", jpeg_quality=95)
                        return opt_bytes
                    elif mode == "cartoon":
                        from ai.cartoon import cartoon_engine
                        from ai.compress import compress_engine
                        style = options.get("cartoon_style", "anime")
                        input_pil = Image.open(io.BytesIO(file_bytes))
                        result_pil, _ = cartoon_engine.stylize_image(input_pil, style=style)
                        opt_bytes, _ = compress_engine.optimize_output(result_pil, target_format="PNG")
                        return opt_bytes
                    elif mode == "object_remove":
                        from ai.object_remove import lama_engine
                        from ai.compress import compress_engine
                        input_pil = Image.open(io.BytesIO(file_bytes))
                        result_pil = lama_engine.remove_object(input_pil)
                        opt_bytes, _ = compress_engine.optimize_output(result_pil, target_format="JPEG", jpeg_quality=95)
                        return opt_bytes

                    else:
                        return file_bytes

                async with self.gpu_semaphore:
                    output_bytes = await asyncio.to_thread(_sync_process)

        finally:
            await ai_scheduler.release_gpu_job()

        # 4. CPU Stage 2: Save Output File (outside GPU lock)
        output_name, output_path = self.storage.save_output_image(output_bytes, filename_prefix="enhanced")

        return {
            "status": "success",
            "upload_name": upload_name,
            "output_name": output_name,
            "output_path": str(output_path),
            "output_url": f"/outputs/{output_name}",
            "processing_time_ms": round(timer.elapsed_ms, 2),
            "options": options,
        }



# Singleton instance
image_service = ImageService()
