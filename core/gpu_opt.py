"""
==============================================================================
Zenemoo AI - GPU Hardware Acceleration & CUDA Performance Flags
==============================================================================
Configures PyTorch CUDA hardware optimization flags for high-speed inference on
NVIDIA RTX GPUs (cuDNN benchmark, TF32 Tensor Cores, float32 precision).
"""

import torch
from core.logging import logger


def setup_gpu_optimizations() -> None:
    """Enables PyTorch CUDA performance flags for maximum RTX 3050 Tensor Core throughput."""
    if torch.cuda.is_available():
        try:
            # Enable cuDNN auto-tuner for optimal convolution kernel selection
            torch.backends.cudnn.benchmark = True
            
            # Enable Ampere TF32 matrix multiplication for RTX 30 Series Tensor Cores
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            
            # Set float32 matrix multiplication precision to 'high' (uses TF32 where supported)
            torch.set_float32_matmul_precision("high")
            
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"⚡ Global GPU CUDA Optimizations Enabled: [{device_name}] (cuDNN Benchmark=True, TF32=True, Matmul Precision=HIGH)")
        except Exception as e:
            logger.warning(f"⚠️ Failed applying some CUDA optimizations: {e}")


# Auto-apply optimizations upon module import
setup_gpu_optimizations()
