"""
==============================================================================
Zenemoo AI - Unified Image Processing Pipeline Orchestrator
==============================================================================
Sequential multi-stage AI enhancement pipeline orchestrating:
Validate -> Background Removal -> Face Restoration -> Real-ESRGAN -> SwinIR Sharpen -> Compress -> Output
Includes Smart Pipeline skipping, Fast Mode tuning, and memory management.
"""

import time
import io
import gc
from typing import Dict, Any, Callable, Optional
from PIL import Image
import torch
from core.logging import logger
from shared.utils.validators import validate_image_upload
from shared.utils.gpu import empty_gpu_cache, get_gpu_info
from shared.utils.image import smart_downscale_pil
from ai.background import bg_remover_engine
from ai.restore import face_restorer_manager
from ai.upscale import realesrgan_engine
from ai.sharpen import sharpen_engine


class UnifiedEnhancementPipeline:
    """Enterprise Multi-Stage AI Pipeline Orchestrator."""

    def _log_stage_telemetry(self, stage_name: str, pil_img: Image.Image, stage_time_s: float) -> None:
        """Logs resolution, system RAM, GPU device, allocated VRAM, peak VRAM, and stage latency."""
        w, h = pil_img.size
        gpu = get_gpu_info()
        logger.info(
            f"⏱️ Stage [{stage_name}] Complete ({stage_time_s:.2f}s) | "
            f"Res: {w}x{h} px ({ (w*h)/1e6:.2f} MP) | "
            f"Device: [{gpu['device_name']}] | "
            f"VRAM: {gpu['allocated_mb']:.1f} MB (Peak: {gpu['max_allocated_mb']:.1f} MB) | "
            f"RAM: {gpu['system_ram_used_mb']:.1f} MB"
        )

    def run_pipeline(
        self,
        image_bytes: bytes,
        filename: str = "input.jpg",
        options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[Callable[[int, str], None]] = None,
    ) -> bytes:
        """
        Executes full sequential AI enhancement chain on input image bytes.
        """
        opts = options or {}
        t_pipeline_start = time.perf_counter()
        
        # Options & feature flags
        fast_mode = opts.get("fast_mode", False)
        remove_bg = opts.get("remove_bg", False)
        face_restore = opts.get("face_restore", True)
        face_model = opts.get("face_model", "gfpgan")
        
        if fast_mode:
            upscale_factor = int(opts.get("upscale_factor", 2))
        else:
            upscale_factor = int(opts.get("upscale_factor", 4))
            
        do_sharpen = opts.get("sharpen", True)
        compress_quality = int(opts.get("quality", 90))

        mode_str = "FAST MODE" if fast_mode else "STANDARD HIGH-QUALITY MODE"
        logger.info(f"🔄 Starting Unified Pipeline for '{filename}' [{mode_str}]...")

        def _notify(percent: int, step_name: str):
            logger.info(f"📊 Pipeline Progress [{percent}%]: {step_name}")
            if progress_callback:
                progress_callback(percent, step_name)

        try:
            # 1. Validation Stage
            _notify(5, "Validating Image Payload")
            validate_image_upload(image_bytes, filename)
            current_pil = Image.open(io.BytesIO(image_bytes))

            # 2. Smart Pre-Resize Stage (Downscale oversized images to 2048px max dim working copy)
            orig_w, orig_h = current_pil.size
            if max(orig_w, orig_h) > 2048:
                _notify(15, f"Downscaling {orig_w}x{orig_h} to 2048px working copy")
                logger.info(f"📐 Input resolution ({orig_w}x{orig_h} px, {(orig_w*orig_h)/1e6:.2f} MP) > 2048px limit. Auto-resizing working copy...")
                current_pil = smart_downscale_pil(current_pil, max_dim=2048)

            # Import image analyzer for smart stage skipping metrics
            from shared.utils.image_analyzer import analyze_image_properties
            analysis = analyze_image_properties(current_pil)

            # 3. Background Removal Stage (Optional)
            if remove_bg:
                t0 = time.perf_counter()
                _notify(30, "Removing Background (rembg U²-Net)")
                current_pil = bg_remover_engine.remove_background(current_pil)
                self._log_stage_telemetry("Background Removal", current_pil, time.perf_counter() - t0)
            else:
                logger.info("ℹ️ Smart Stage Skipping: Background removal not requested. Skipping rembg stage.")

            # 4. Smart Face Restoration Stage (GFPGAN / CodeFormer)
            if face_restore:
                if analysis["has_faces"]:
                    t0 = time.perf_counter()
                    _notify(50, f"Restoring Faces ({face_model.upper()})")
                    current_pil = face_restorer_manager.restore_face(
                        current_pil,
                        model=face_model,
                        fidelity=0.7,
                        only_center_face=fast_mode,
                    )
                    self._log_stage_telemetry("Face Restoration", current_pil, time.perf_counter() - t0)
                else:
                    logger.info("ℹ️ Smart Stage Skipping: 0 faces detected in input image. Skipping face restoration stage.")

            # 5. Super Resolution Stage (Real-ESRGAN)
            force_upscale = opts.get("force_upscale", False)
            curr_w, curr_h = current_pil.size
            is_already_high_res = (max(curr_w, curr_h) >= 2000 or (curr_w * curr_h) / 1e6 >= 4.0)
            
            if upscale_factor in [2, 4]:
                if is_already_high_res and not force_upscale:
                    logger.info(f"ℹ️ Smart Stage Skipping: Image resolution ({curr_w}x{curr_h} px) is already high. Skipping Real-ESRGAN upscaling stage.")
                else:
                    t0 = time.perf_counter()
                    _notify(70, f"Super Resolution Upscaling ({upscale_factor}x Real-ESRGAN)")
                    current_pil = realesrgan_engine.upscale(
                        current_pil,
                        scale=upscale_factor,
                        fast_mode=fast_mode,
                    )
                    self._log_stage_telemetry("Real-ESRGAN Upscale", current_pil, time.perf_counter() - t0)

            # 6. Denoise & Sharpen Stage (SwinIR)
            if do_sharpen:
                if analysis["is_sharp"]:
                    logger.info(f"ℹ️ Smart Stage Skipping: Image is already sharp (Laplacian variance {analysis['laplacian_variance']} > 300). Skipping SwinIR sharpen stage.")
                else:
                    t0 = time.perf_counter()
                    _notify(85, "Sharpening & Denoising (SwinIR)")
                    current_pil = sharpen_engine.sharpen_and_denoise(current_pil)
                    self._log_stage_telemetry("SwinIR Sharpen", current_pil, time.perf_counter() - t0)


            # 7. Smart Compression & Output Optimization Stage
            _notify(95, "Optimizing Image Compression & Output Formatting")
            from ai.compress import compress_engine
            target_fmt = opts.get("output_format") or ("PNG" if remove_bg else None)
            webp_lossless = opts.get("webp_lossless", False)

            output_bytes, telemetry = compress_engine.optimize_output(
                image_input=current_pil,
                target_format=target_fmt,
                jpeg_quality=compress_quality,
                webp_quality=compress_quality,
                webp_lossless=webp_lossless,
            )

            t_total = time.perf_counter() - t_pipeline_start
            final_w, final_h = current_pil.size
            logger.info(f"✨ UNIFIED PIPELINE COMPLETE in {t_total:.2f}s | Output: {final_w}x{final_h} px ({len(output_bytes)/1024:.1f} KB)")

            _notify(100, "Pipeline Execution Complete")
            return output_bytes


        finally:
            empty_gpu_cache()
            gc.collect()


# Singleton Instance
unified_pipeline = UnifiedEnhancementPipeline()
