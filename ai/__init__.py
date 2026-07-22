"""
==============================================================================
Zenemoo AI Engine Package & Model Warmup Manager
==============================================================================
Pre-loads all 6 AI deep learning model engines ONCE into CUDA memory on startup
so subsequent image requests reuse already initialized models with 0 reload latency.
"""

import time
import torch
from core.logging import logger
from core.gpu_opt import setup_gpu_optimizations


def warmup_all_models() -> None:
    """Loads all 6 AI engines into CUDA VRAM once on application startup."""
    setup_gpu_optimizations()
    logger.info("🚀 Pre-loading and warming up all 6 AI deep learning model engines on CUDA...")
    t0 = time.perf_counter()

    try:
        from ai.background import bg_remover_engine
        from ai.restore import gfpgan_engine, codeformer_engine
        from ai.upscale import realesrgan_engine
        from ai.sharpen import sharpen_engine
        from ai.object_remove import lama_engine

        # Initialize background remover ONNX session
        logger.info("  [1/6] Warming up rembg (U²-Net)...")
        bg_remover_engine._get_session()

        # Initialize GFPGAN face restoration model
        logger.info("  [2/6] Warming up GFPGAN v1.4...")
        gfpgan_engine._init_gfpgan()

        # Initialize Real-ESRGAN super resolution models
        logger.info("  [3/6] Warming up Real-ESRGAN (4x & 2x)...")
        realesrgan_engine._init_model(scale=4)

        # Initialize CodeFormer face restoration model
        logger.info("  [4/6] Warming up CodeFormer...")
        codeformer_engine._init_codeformer()

        # Initialize SwinIR sharpen model
        logger.info("  [5/6] Warming up SwinIR Transformer...")
        sharpen_engine._init_swinir()

        # Initialize LaMa object removal engine
        logger.info("  [6/6] Warming up LaMa Inpainting...")
        lama_engine._init_lama()

        elapsed = (time.perf_counter() - t0) * 1000
        if torch.cuda.is_available():
            vram_mb = torch.cuda.memory_allocated(0) / (1024 * 1024)
            logger.info(f"✨ All 6 AI engines pre-loaded successfully on CUDA in {elapsed:.2f} ms | Total VRAM Allocated: {vram_mb:.2f} MB")
        else:
            logger.info(f"✨ All 6 AI engines pre-loaded successfully in {elapsed:.2f} ms")

    except Exception as e:
        logger.error(f"⚠️ Error during AI model warmup: {e}", exc_info=True)


__all__ = ["warmup_all_models"]
