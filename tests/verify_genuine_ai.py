"""
==============================================================================
Zenemoo AI - Senior AI Performance Engineer Benchmark & Profiling Suite
==============================================================================
Profiles end-to-end deep learning inference across all 6 AI model engines on
NVIDIA GeForce RTX 3050 (6GB VRAM), measuring stage timings, peak VRAM/RAM,
precision, tile sizes, and pipeline throughput.
"""

import sys
import time
import io
from pathlib import Path
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from core.logging import logger
from core.gpu_opt import setup_gpu_optimizations
from shared.utils.gpu import get_gpu_info, reset_gpu_peak_memory, empty_gpu_cache
from ai import warmup_all_models
from ai.background import bg_remover_engine
from ai.restore import gfpgan_engine, codeformer_engine
from ai.upscale import realesrgan_engine
from ai.sharpen import sharpen_engine
from ai.object_remove import lama_engine
from ai.enhancer import unified_pipeline


def create_test_image_bytes(width: int = 1920, height: int = 1080) -> bytes:
    """Creates a sample test image at requested resolution."""
    img = Image.new("RGB", (width, height), color=(120, 160, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def run_benchmark():
    setup_gpu_optimizations()
    logger.info("================================================================================")
    logger.info("⚡ ZENEMOO AI - SENIOR AI PERFORMANCE ENGINEER BENCHMARK REPORT ⚡")
    logger.info("================================================================================")

    # 1. Warmup Model Cache (Load ONCE)
    warmup_all_models()
    reset_gpu_peak_memory()

    # Hardware & Telemetry
    gpu = get_gpu_info()
    precision = "FP16 (Half)" if torch.cuda.is_available() else "FP32 (Float)"
    logger.info(f"🖥️ Target GPU: [{gpu['device_name']}] | Available VRAM: {gpu['total_vram_mb']:.2f} MB")
    logger.info(f"⚙️ CUDA Config: FP16 Precision={precision} | cuDNN Benchmark=True | TF32=True")
    logger.info("--------------------------------------------------------------------------------")

    # Benchmarks to run
    test_resolutions = [
        ("720p HD", 1280, 720),
        ("1080p Full HD", 1920, 1080),
    ]

    for label, w, h in test_resolutions:
        mp = (w * h) / 1_000_000.0
        logger.info(f"\n🚀 BENCHMARKING IMAGE RESOLUTION: {label} ({w}x{h} px, {mp:.2f} MP)")
        logger.info("-" * 80)
        
        input_bytes = create_test_image_bytes(w, h)
        test_pil = Image.open(io.BytesIO(input_bytes))

        reset_gpu_peak_memory()

        # 1. Stage: rembg
        t0 = time.perf_counter()
        res_bg = bg_remover_engine.remove_background(test_pil)
        t_bg = time.perf_counter() - t0

        # 2. Stage: GFPGAN
        t0 = time.perf_counter()
        res_gfp = gfpgan_engine.restore(test_pil)
        t_gfp = time.perf_counter() - t0

        # 3. Stage: Real-ESRGAN (4x)
        t0 = time.perf_counter()
        res_sr = realesrgan_engine.upscale(test_pil, scale=4)
        t_sr = time.perf_counter() - t0

        # 4. Stage: SwinIR
        t0 = time.perf_counter()
        res_swin = sharpen_engine.sharpen_and_denoise(test_pil)
        t_swin = time.perf_counter() - t0

        # 5. Stage: CodeFormer
        t0 = time.perf_counter()
        res_code = codeformer_engine.restore(test_pil, fidelity=0.7)
        t_code = time.perf_counter() - t0

        # 6. Stage: LaMa Inpainting
        t0 = time.perf_counter()
        res_lama = lama_engine.remove_object(test_pil)
        t_lama = time.perf_counter() - t0

        # 7. Complete Unified Pipeline (Fast Mode)
        t0 = time.perf_counter()
        res_pipe_fast = unified_pipeline.run_pipeline(
            image_bytes=input_bytes,
            filename=f"test_{label}.jpg",
            options={
                "fast_mode": True,
                "remove_bg": False,
                "face_restore": True,
                "upscale_factor": 2,
                "sharpen": True,
            }
        )
        t_pipe_fast = time.perf_counter() - t0

        # 8. Complete Unified Pipeline (Full HQ Mode)
        t0 = time.perf_counter()
        res_pipe_hq = unified_pipeline.run_pipeline(
            image_bytes=input_bytes,
            filename=f"test_{label}.jpg",
            options={
                "fast_mode": False,
                "remove_bg": True,
                "face_restore": True,
                "upscale_factor": 4,
                "sharpen": True,
            }
        )
        t_pipe_hq = time.perf_counter() - t0

        # Post-stage telemetry
        post_gpu = get_gpu_info()
        fps_hq = 1.0 / t_pipe_hq if t_pipe_hq > 0 else 0.0
        fps_fast = 1.0 / t_pipe_fast if t_pipe_fast > 0 else 0.0

        # Print Benchmark Report Table
        print("\n" + "=" * 80)
        print(f"📊 PERFORMANCE BENCHMARK REPORT [{label} - {w}x{h} ({mp:.2f} MP)]")
        print("=" * 80)
        print(f" • GPU Device:           {post_gpu['device_name']}")
        print(f" • Precision:              {precision}")
        print(f" • Peak GPU VRAM:          {post_gpu['max_allocated_mb']:.2f} MB / {post_gpu['total_vram_mb']:.2f} MB")
        print(f" • VRAM Remaining:         {post_gpu['remaining_vram_mb']:.2f} MB")
        print(f" • Peak System RAM:        {post_gpu['system_ram_used_mb']:.2f} MB / {post_gpu['system_ram_total_mb']:.2f} MB")
        print("-" * 80)
        print(" INDIVIDUAL STAGE EXECUTION TIMINGS:")
        print(f"   1. rembg (U²-Net):           {t_bg:6.2f} s")
        print(f"   2. GFPGAN (v1.4 Face):        {t_gfp:6.2f} s")
        print(f"   3. Real-ESRGAN (4x SuperRes): {t_sr:6.2f} s")
        print(f"   4. SwinIR (Sharpen/Denoise):  {t_swin:6.2f} s")
        print(f"   5. CodeFormer (Transformer):  {t_code:6.2f} s")
        print(f"   6. LaMa (FFC Inpainting):     {t_lama:6.2f} s")
        print("-" * 80)
        print(f" 🚀 FAST MODE PIPELINE TOTAL:   {t_pipe_fast:6.2f} s (FPS: {fps_fast:.2f})")
        print(f" 🎨 FULL HQ PIPELINE TOTAL:     {t_pipe_hq:6.2f} s (FPS: {fps_hq:.2f})")
        print("=" * 80 + "\n")

    logger.info("🏆 ALL PERFORMANCE BENCHMARKS AND OPTIMIZATION CHECKS COMPLETED!")


if __name__ == "__main__":
    run_benchmark()
